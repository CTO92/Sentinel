/// @file platform_detect.h
/// @brief Runtime GPU platform detection (CUDA vs ROCm/HIP).
///
/// Probes available drivers and libraries to determine which GPU
/// runtime is available and populates a PlatformInfo struct used
/// throughout the agent.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sentinel::platform {

/// GPU vendor / runtime type.
enum class GpuRuntime : uint8_t {
    kNone = 0,
    kCuda = 1,
    kHip  = 2,
};

/// Information about a single detected GPU device.
struct GpuDeviceInfo {
    /// Device index (0-based).
    int device_index = -1;

    /// PCI bus ID string.
    std::string pci_bus_id;

    /// UUID string (e.g., "GPU-xxxxxxxx-...").
    std::string uuid;

    /// Human-readable model name.
    std::string name;

    /// Total global memory in bytes.
    std::size_t total_memory = 0;

    /// Number of streaming multiprocessors (CUs on AMD).
    int sm_count = 0;

    /// Compute capability major version.
    int compute_major = 0;

    /// Compute capability minor version.
    int compute_minor = 0;

    /// Maximum resident threads per SM.
    int max_threads_per_sm = 0;

    /// Maximum shared memory per SM in bytes.
    std::size_t shared_memory_per_sm = 0;

    /// Maximum number of resident blocks per SM.
    int max_blocks_per_sm = 0;
};

/// Aggregated platform information.
struct PlatformInfo {
    /// Detected runtime.
    GpuRuntime runtime = GpuRuntime::kNone;

    /// Driver version string.
    std::string driver_version;

    /// Runtime version string.
    std::string runtime_version;

    /// List of detected GPU devices.
    std::vector<GpuDeviceInfo> devices;
};

/// Detect the available GPU platform and enumerate devices.
/// Returns a PlatformInfo with runtime == kNone if no GPUs are found.
[[nodiscard]] PlatformInfo detect_platform();

/// Get a human-readable string for a GpuRuntime value.
[[nodiscard]] const char* runtime_to_string(GpuRuntime rt);

}  // namespace sentinel::platform
