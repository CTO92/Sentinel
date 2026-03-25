/**
 * SENTINEL SDC Injector — Controlled Silent Data Corruption Injection
 *
 * This module provides CUDA kernels for deterministic injection of various
 * fault types into GPU memory and execution units.  Every injection is logged
 * to stdout in structured JSON so that the SENTINEL detection pipeline can be
 * validated end-to-end.
 *
 * SAFETY: The injector is a no-op unless the caller passes the
 *         --enable-injection flag (CLI) or sets enable_injection=true
 *         (library API).  This prevents accidental corruption of production
 *         workloads.
 *
 * Copyright 2025-2026 SENTINEL Authors — Apache 2.0
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <curand_kernel.h>

/* ======================================================================== */
/*  Compile-time configuration                                              */
/* ======================================================================== */
#ifndef SDC_INJECTOR_MAX_LOG_QUEUE
#define SDC_INJECTOR_MAX_LOG_QUEUE 4096
#endif

/* ======================================================================== */
/*  Structured JSON logger                                                  */
/* ======================================================================== */

static void log_json(const char* event_type,
                     const char* target,
                     const char* detail_fmt, ...) {
    char detail[1024];
    va_list ap;
    va_start(ap, detail_fmt);
    vsnprintf(detail, sizeof(detail), detail_fmt, ap);
    va_end(ap);

    /* ISO-8601 timestamp */
    time_t now = time(nullptr);
    struct tm tm_buf;
#ifdef _WIN32
    gmtime_s(&tm_buf, &now);
#else
    gmtime_r(&now, &tm_buf);
#endif
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);

    fprintf(stdout,
            "{\"timestamp\":\"%s\",\"component\":\"sdc-injector\","
            "\"event\":\"%s\",\"target\":\"%s\",\"details\":{%s}}\n",
            ts, event_type, target, detail);
    fflush(stdout);
}

/* ======================================================================== */
/*  Error checking helper                                                   */
/* ======================================================================== */

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            log_json("error", "cuda",                                           \
                     "\"function\":\"%s\",\"code\":%d,\"message\":\"%s\"",      \
                     #call, (int)err, cudaGetErrorString(err));                 \
            return -1;                                                          \
        }                                                                       \
    } while (0)

/* ======================================================================== */
/*  Global safety flag                                                       */
/* ======================================================================== */

static bool g_injection_enabled = false;

extern "C" void sdc_injector_enable(bool enable) {
    g_injection_enabled = enable;
    log_json("config", "safety",
             "\"injection_enabled\":%s", enable ? "true" : "false");
}

extern "C" bool sdc_injector_is_enabled(void) {
    return g_injection_enabled;
}

/* ======================================================================== */
/*  Kernel: bit-flip injection in global memory                             */
/* ======================================================================== */

/**
 * Flip a single bit at a given byte offset and bit position inside a
 * device buffer.  The kernel is launched with a single thread.
 */
__global__ void kernel_inject_bitflip(uint8_t* base,
                                       size_t   byte_offset,
                                       int      bit_position) {
    uint8_t mask = 1u << bit_position;
    uint8_t old_val = base[byte_offset];
    base[byte_offset] = old_val ^ mask;
    /* Store old/new for host-side logging (via mapped memory if needed). */
    printf("{\"kernel\":\"bitflip\",\"offset\":%llu,\"bit\":%d,"
           "\"old\":\"0x%02x\",\"new\":\"0x%02x\"}\n",
           (unsigned long long)byte_offset, bit_position,
           (unsigned)old_val, (unsigned)(old_val ^ mask));
}

/**
 * Host wrapper.  Returns 0 on success, -1 on error.
 */
extern "C" int inject_bitflip(void* device_address,
                               size_t byte_offset,
                               int    bit_position) {
    if (!g_injection_enabled) {
        log_json("rejected", "bitflip",
                 "\"reason\":\"injection not enabled\"");
        return -1;
    }
    if (bit_position < 0 || bit_position > 7) {
        log_json("error", "bitflip",
                 "\"reason\":\"bit_position must be 0-7\",\"got\":%d",
                 bit_position);
        return -1;
    }

    log_json("inject", "bitflip",
             "\"address\":\"0x%llx\",\"byte_offset\":%llu,\"bit_position\":%d",
             (unsigned long long)device_address,
             (unsigned long long)byte_offset,
             bit_position);

    kernel_inject_bitflip<<<1, 1>>>(
        reinterpret_cast<uint8_t*>(device_address), byte_offset, bit_position);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ======================================================================== */
/*  Kernel: stuck-at fault on a specific SM                                 */
/* ======================================================================== */

/**
 * Every thread on the targeted SM overwrites its FMA result with a fixed
 * constant.  Threads on other SMs are no-ops.  The kernel is designed to
 * be launched over many SMs so that only the target SM produces the fault.
 *
 * Uses inline PTX to read %smid.
 */
__global__ void kernel_inject_stuck_at(float*   output,
                                        int      target_sm,
                                        float    stuck_value,
                                        int      count) {
    unsigned int smid;
    asm("mov.u32 %0, %%smid;" : "=r"(smid));

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if ((int)smid == target_sm) {
        /* Replace the value with the stuck constant. */
        output[idx] = stuck_value;
    }
    /* else: leave output unchanged (no-op) */
}

extern "C" int inject_stuck_at(float* device_output,
                                int    count,
                                int    target_sm,
                                float  stuck_value) {
    if (!g_injection_enabled) {
        log_json("rejected", "stuck_at",
                 "\"reason\":\"injection not enabled\"");
        return -1;
    }

    log_json("inject", "stuck_at",
             "\"target_sm\":%d,\"stuck_value\":%f,\"count\":%d",
             target_sm, stuck_value, count);

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    kernel_inject_stuck_at<<<blocks, threads>>>(
        device_output, target_sm, stuck_value, count);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ======================================================================== */
/*  Kernel: Gaussian noise injection                                        */
/* ======================================================================== */

__global__ void kernel_inject_noise(float*       data,
                                     int          count,
                                     float        sigma,
                                     unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    float noise = curand_normal(&state) * sigma;
    data[idx] += noise;
}

extern "C" int inject_noise(float* device_address,
                             int    count,
                             float  sigma) {
    if (!g_injection_enabled) {
        log_json("rejected", "noise",
                 "\"reason\":\"injection not enabled\"");
        return -1;
    }

    unsigned long long seed = (unsigned long long)time(nullptr) ^ 0xDEADBEEFULL;

    log_json("inject", "noise",
             "\"address\":\"0x%llx\",\"count\":%d,\"sigma\":%f,\"seed\":%llu",
             (unsigned long long)device_address, count, sigma, seed);

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    kernel_inject_noise<<<blocks, threads>>>(
        device_address, count, sigma, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ======================================================================== */
/*  Kernel: shared memory corruption                                        */
/* ======================================================================== */

/**
 * Allocate dynamic shared memory and corrupt a specific word.
 * Useful for testing probes that validate shared-memory integrity.
 */
__global__ void kernel_corrupt_shared(float*  output,
                                       int     count,
                                       int     smem_word_idx,
                                       float   corrupt_value) {
    extern __shared__ float smem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    /* Cooperative load into shared memory */
    smem[threadIdx.x] = output[idx];
    __syncthreads();

    /* Corrupt the target word in every block */
    if (threadIdx.x == 0 && smem_word_idx < blockDim.x) {
        smem[smem_word_idx] = corrupt_value;
    }
    __syncthreads();

    /* Write back — the corrupted word propagates. */
    output[idx] = smem[threadIdx.x];
}

extern "C" int inject_shared_memory_corruption(float* device_output,
                                                int    count,
                                                int    smem_word_idx,
                                                float  corrupt_value) {
    if (!g_injection_enabled) {
        log_json("rejected", "shared_memory",
                 "\"reason\":\"injection not enabled\"");
        return -1;
    }

    log_json("inject", "shared_memory",
             "\"smem_word_idx\":%d,\"corrupt_value\":%f,\"count\":%d",
             smem_word_idx, corrupt_value, count);

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    size_t smem_size = threads * sizeof(float);
    kernel_corrupt_shared<<<blocks, threads, smem_size>>>(
        device_output, count, smem_word_idx, corrupt_value);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ======================================================================== */
/*  Kernel: register corruption via inline PTX                              */
/* ======================================================================== */

/**
 * XOR a specific 32-bit pattern into the first FP register value used by
 * the thread.  This simulates a transient register fault.
 */
__global__ void kernel_corrupt_register(float*   output,
                                         int      count,
                                         uint32_t xor_mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    float val = output[idx];
    uint32_t bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(val));
    bits ^= xor_mask;
    asm("mov.b32 %0, %1;" : "=f"(val) : "r"(bits));
    output[idx] = val;
}

extern "C" int inject_register_corruption(float*   device_output,
                                           int      count,
                                           uint32_t xor_mask) {
    if (!g_injection_enabled) {
        log_json("rejected", "register",
                 "\"reason\":\"injection not enabled\"");
        return -1;
    }

    log_json("inject", "register",
             "\"xor_mask\":\"0x%08x\",\"count\":%d",
             xor_mask, count);

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    kernel_corrupt_register<<<blocks, threads>>>(
        device_output, count, xor_mask);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ======================================================================== */
/*  Kernel: tensor core (HMMA) output corruption                            */
/* ======================================================================== */

/**
 * Corrupt the output of a simulated tensor-core HMMA operation by
 * injecting a bit-flip into every element of the output matrix fragment.
 * The kernel operates on half-precision (FP16) data stored as uint16_t.
 */
__global__ void kernel_corrupt_tensor_core(uint16_t* output,
                                            int       count,
                                            int       bit_position) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint16_t mask = (uint16_t)(1u << bit_position);
    output[idx] ^= mask;
}

extern "C" int inject_tensor_core_corruption(void* device_output,
                                              int   count,
                                              int   bit_position) {
    if (!g_injection_enabled) {
        log_json("rejected", "tensor_core",
                 "\"reason\":\"injection not enabled\"");
        return -1;
    }
    if (bit_position < 0 || bit_position > 15) {
        log_json("error", "tensor_core",
                 "\"reason\":\"bit_position must be 0-15 for FP16\"");
        return -1;
    }

    log_json("inject", "tensor_core",
             "\"count\":%d,\"bit_position\":%d", count, bit_position);

    int threads = 256;
    int blocks  = (count + threads - 1) / threads;
    kernel_corrupt_tensor_core<<<blocks, threads>>>(
        reinterpret_cast<uint16_t*>(device_output), count, bit_position);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ======================================================================== */
/*  Kernel: memory stuck-bit simulation                                     */
/* ======================================================================== */

/**
 * Simulate a stuck bit in global memory at a specific byte offset.
 * The stuck bit is forced to `stuck_value` (0 or 1) on every write.
 */
__global__ void kernel_memory_stuck_bit(uint8_t* base,
                                         size_t   byte_offset,
                                         int      bit_position,
                                         int      stuck_value) {
    uint8_t mask = (uint8_t)(1u << bit_position);
    if (stuck_value) {
        base[byte_offset] |= mask;   /* stuck-at-1 */
    } else {
        base[byte_offset] &= ~mask;  /* stuck-at-0 */
    }
}

extern "C" int inject_memory_stuck_bit(void*  device_address,
                                        size_t byte_offset,
                                        int    bit_position,
                                        int    stuck_value) {
    if (!g_injection_enabled) {
        log_json("rejected", "memory_stuck_bit",
                 "\"reason\":\"injection not enabled\"");
        return -1;
    }

    log_json("inject", "memory_stuck_bit",
             "\"address\":\"0x%llx\",\"byte_offset\":%llu,"
             "\"bit_position\":%d,\"stuck_value\":%d",
             (unsigned long long)device_address,
             (unsigned long long)byte_offset,
             bit_position, stuck_value);

    kernel_memory_stuck_bit<<<1, 1>>>(
        reinterpret_cast<uint8_t*>(device_address),
        byte_offset, bit_position, stuck_value);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

/* ======================================================================== */
/*  CLI entry point (only when built as SDC_INJECTOR_BUILD_CLI)             */
/* ======================================================================== */

#ifdef SDC_INJECTOR_BUILD_CLI

static void print_usage(const char* prog) {
    fprintf(stderr,
        "SENTINEL SDC Injector\n"
        "Usage: %s --enable-injection <command> [options]\n\n"
        "Commands:\n"
        "  bitflip     --offset <N> --bit <0-7>          Flip one bit in device memory\n"
        "  stuck-at    --sm <id> --value <f>  --count <N> Stuck-at fault on SM\n"
        "  noise       --count <N> --sigma <f>            Gaussian noise injection\n"
        "  stuck-bit   --offset <N> --bit <0-7> --value <0|1>  Stuck memory bit\n"
        "  tensor-core --count <N> --bit <0-15>           Corrupt HMMA output\n"
        "  register    --count <N> --mask <hex>           Register XOR corruption\n"
        "  shared-mem  --count <N> --word <idx> --value <f>  Shared memory corruption\n"
        "  selftest                                       Allocate memory and run all types\n\n"
        "SAFETY: --enable-injection is REQUIRED.\n",
        prog);
}

static int run_selftest() {
    log_json("selftest", "start", "\"message\":\"running built-in self-test\"");

    const int N = 1024;
    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_buf, 0, N * sizeof(float)));

    /* Bit-flip */
    if (inject_bitflip(d_buf, 0, 3) != 0) return 1;

    /* Stuck-at (SM 0, value 42.0) */
    if (inject_stuck_at(d_buf, N, 0, 42.0f) != 0) return 1;

    /* Noise */
    if (inject_noise(d_buf, N, 0.01f) != 0) return 1;

    /* Memory stuck bit */
    if (inject_memory_stuck_bit(d_buf, 4, 5, 1) != 0) return 1;

    /* Tensor core (treat as FP16 buffer, use first 512 elements) */
    if (inject_tensor_core_corruption(d_buf, 512, 7) != 0) return 1;

    /* Register corruption */
    if (inject_register_corruption(d_buf, N, 0x00000100) != 0) return 1;

    /* Shared memory corruption */
    if (inject_shared_memory_corruption(d_buf, N, 42, 999.0f) != 0) return 1;

    CUDA_CHECK(cudaFree(d_buf));

    log_json("selftest", "complete",
             "\"message\":\"all injection types executed successfully\"");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) { print_usage(argv[0]); return 1; }

    /* Check for --enable-injection flag */
    bool found_enable = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--enable-injection") == 0) {
            found_enable = true;
            break;
        }
    }
    if (!found_enable) {
        fprintf(stderr, "ERROR: --enable-injection flag is REQUIRED.\n");
        return 1;
    }
    sdc_injector_enable(true);

    /* Find the command argument (first non-flag argument after program name) */
    const char* command = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--enable-injection") == 0) continue;
        if (argv[i][0] == '-') continue;
        command = argv[i];
        break;
    }
    if (!command) { print_usage(argv[0]); return 1; }

    /* Helper to parse named args */
    auto get_arg = [&](const char* name) -> const char* {
        for (int i = 1; i < argc - 1; ++i) {
            if (strcmp(argv[i], name) == 0) return argv[i + 1];
        }
        return nullptr;
    };

    if (strcmp(command, "selftest") == 0) {
        return run_selftest();
    }

    /* For all other commands we need a device buffer. Allocate a default. */
    const int default_count = 1024;
    int count = default_count;
    const char* count_str = get_arg("--count");
    if (count_str) count = atoi(count_str);

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, count * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_buf, 0, count * sizeof(float)));

    int rc = 0;
    if (strcmp(command, "bitflip") == 0) {
        int offset = 0, bit = 0;
        const char* v;
        if ((v = get_arg("--offset"))) offset = atoi(v);
        if ((v = get_arg("--bit")))    bit    = atoi(v);
        rc = inject_bitflip(d_buf, offset, bit);

    } else if (strcmp(command, "stuck-at") == 0) {
        int sm = 0;
        float val = 0.0f;
        const char* v;
        if ((v = get_arg("--sm")))    sm  = atoi(v);
        if ((v = get_arg("--value"))) val = (float)atof(v);
        rc = inject_stuck_at(d_buf, count, sm, val);

    } else if (strcmp(command, "noise") == 0) {
        float sigma = 0.01f;
        const char* v;
        if ((v = get_arg("--sigma"))) sigma = (float)atof(v);
        rc = inject_noise(d_buf, count, sigma);

    } else if (strcmp(command, "stuck-bit") == 0) {
        int offset = 0, bit = 0, val = 1;
        const char* v;
        if ((v = get_arg("--offset"))) offset = atoi(v);
        if ((v = get_arg("--bit")))    bit    = atoi(v);
        if ((v = get_arg("--value")))  val    = atoi(v);
        rc = inject_memory_stuck_bit(d_buf, offset, bit, val);

    } else if (strcmp(command, "tensor-core") == 0) {
        int bit = 7;
        const char* v;
        if ((v = get_arg("--bit"))) bit = atoi(v);
        rc = inject_tensor_core_corruption(d_buf, count, bit);

    } else if (strcmp(command, "register") == 0) {
        uint32_t mask = 0x00000100;
        const char* v;
        if ((v = get_arg("--mask"))) mask = (uint32_t)strtoul(v, nullptr, 16);
        rc = inject_register_corruption(d_buf, count, mask);

    } else if (strcmp(command, "shared-mem") == 0) {
        int word = 0;
        float val = 0.0f;
        const char* v;
        if ((v = get_arg("--word")))  word = atoi(v);
        if ((v = get_arg("--value"))) val  = (float)atof(v);
        rc = inject_shared_memory_corruption(d_buf, count, word, val);

    } else {
        fprintf(stderr, "Unknown command: %s\n", command);
        print_usage(argv[0]);
        rc = 1;
    }

    CUDA_CHECK(cudaFree(d_buf));
    return rc;
}

#endif /* SDC_INJECTOR_BUILD_CLI */
