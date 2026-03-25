/// @file hip_runtime.cpp
/// @brief HIP/ROCm runtime abstraction — platform detection and SM affinity for AMD.
///
/// Provides AMD-specific implementations for Compute Unit (CU) affinity,
/// device enumeration, and runtime initialization that mirrors the CUDA
/// runtime layer.

#include "platform/hip_runtime.h"
#include "util/logging.h"

#ifdef SENTINEL_ENABLE_ROCM

#include <hip/hip_runtime.h>
#include <algorithm>
#include <random>

namespace sentinel::platform {

namespace {

/// Validate that a device index is within range.
bool validate_device_index(int device_index) {
    int count = get_device_count_hip();
    if (device_index < 0 || device_index >= count) {
        SENTINEL_LOG_ERROR("Invalid HIP device index {}: {} devices available",
                           device_index, count);
        return false;
    }
    return true;
}

}  // namespace

// ---- Compute Unit Affinity ----
//
// AMD GPUs use Compute Units (CUs) rather than Streaming Multiprocessors.
// HIP does not provide a direct equivalent of CUDA's `%smid` PTX register,
// but on GCN/CDNA architectures the hardware CU ID can be read via
// inline assembly: `s_getreg_b32 <dst>, hwreg(HW_REG_HW_ID, 8, 4)` on
// GCN, or `s_getreg_b32 <dst>, hwreg(HW_REG_HW_ID2, 0, 11)` on CDNA2+.
//
// For probe pinning, we use the cooperative groups grid approach:
// launch enough blocks to fill all CUs, then filter in-kernel based on
// the runtime CU ID — the same strategy used in the CUDA probe agent
// with `%smid`.

/// Query the number of Compute Units on a device.
int get_cu_count(int device_index) {
    if (!validate_device_index(device_index)) return 0;
    hipDeviceProp_t props = {};
    HIP_CHECK(hipGetDeviceProperties(&props, device_index));
    return props.multiProcessorCount;
}

/// Compute launch parameters to ensure full CU coverage.
/// Returned as {blocks, threads_per_block}.
std::pair<int, int> compute_cu_coverage_launch_params(int device_index, int threads_per_block) {
    int cu_count = get_cu_count(device_index);
    if (cu_count == 0) return {0, 0};

    // Over-subscribe by 2x to ensure all CUs get at least one block,
    // even with scheduling imbalances.
    int blocks = cu_count * 2;
    return {blocks, threads_per_block};
}

/// Select a random subset of CU indices for targeted probing.
std::vector<int> select_random_cus(int device_index, float fraction) {
    int cu_count = get_cu_count(device_index);
    if (cu_count == 0) return {};

    int target = std::max(1, static_cast<int>(cu_count * fraction));
    target = std::min(target, cu_count);

    std::vector<int> all_cus(cu_count);
    for (int i = 0; i < cu_count; ++i) all_cus[i] = i;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(all_cus.begin(), all_cus.end(), gen);
    all_cus.resize(target);
    std::sort(all_cus.begin(), all_cus.end());
    return all_cus;
}

/// Log device info for diagnostics.
void log_device_info(int device_index) {
    auto info = get_device_info_hip(device_index);
    SENTINEL_LOG_INFO(
        "HIP Device {}: {} ({}), {} CUs, {:.1f} GB VRAM, wavefront={}",
        info.device_index,
        info.name,
        info.gcn_arch_name,
        info.compute_units,
        static_cast<double>(info.global_mem_bytes) / (1024.0 * 1024.0 * 1024.0),
        info.warp_size
    );
}

/// Initialize HIP runtime and log all available devices.
bool initialize_hip_runtime() {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess || count == 0) {
        SENTINEL_LOG_WARN("No HIP/ROCm devices found (error: {})",
                          hipGetErrorString(err));
        return false;
    }

    SENTINEL_LOG_INFO("HIP runtime initialized: {} device(s) found", count);
    for (int i = 0; i < count; ++i) {
        log_device_info(i);
    }
    return true;
}

}  // namespace sentinel::platform

#else  // !SENTINEL_ENABLE_ROCM

// No-op when ROCm is not enabled.
// The header-only stubs in hip_runtime.h handle the fallback.

#endif  // SENTINEL_ENABLE_ROCM
