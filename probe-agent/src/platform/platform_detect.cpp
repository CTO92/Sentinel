/// @file platform_detect.cpp
/// @brief Runtime GPU platform detection implementation.

#include "platform/platform_detect.h"
#include "util/logging.h"

#include <cuda_runtime.h>

namespace sentinel::platform {

namespace {

/// Attempt to enumerate CUDA devices and fill PlatformInfo.
bool detect_cuda(PlatformInfo& info) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    int runtime_ver = 0;
    cudaRuntimeGetVersion(&runtime_ver);
    info.runtime_version = std::to_string(runtime_ver / 1000) + "." +
                           std::to_string((runtime_ver % 1000) / 10);

    int driver_ver = 0;
    cudaDriverGetVersion(&driver_ver);
    info.driver_version = std::to_string(driver_ver / 1000) + "." +
                          std::to_string((driver_ver % 1000) / 10);

    info.runtime = GpuRuntime::kCuda;
    info.devices.reserve(static_cast<std::size_t>(device_count));

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop{};
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            SENTINEL_LOG_WARN("Failed to get properties for CUDA device {}: {}",
                              i, cudaGetErrorString(err));
            continue;
        }

        GpuDeviceInfo dev;
        dev.device_index = i;
        dev.name = prop.name;
        dev.total_memory = prop.totalGlobalMem;
        dev.sm_count = prop.multiProcessorCount;
        dev.compute_major = prop.major;
        dev.compute_minor = prop.minor;
        dev.max_threads_per_sm = prop.maxThreadsPerMultiProcessor;
        dev.shared_memory_per_sm = prop.sharedMemPerMultiprocessor;
        dev.max_blocks_per_sm = prop.maxBlocksPerMultiProcessor;

        // Get PCI bus ID string.
        char pci_bus[32] = {};
        cudaDeviceGetPCIBusId(pci_bus, sizeof(pci_bus), i);
        dev.pci_bus_id = pci_bus;

        // Get UUID from cudaDeviceProp.uuid (available since CUDA 9.2).
        {
            const auto& uuid = prop.uuid;
            char uuid_str[48];
            std::snprintf(uuid_str, sizeof(uuid_str),
                         "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-"
                         "%02x%02x-%02x%02x%02x%02x%02x%02x",
                         static_cast<uint8_t>(uuid.bytes[0]),
                         static_cast<uint8_t>(uuid.bytes[1]),
                         static_cast<uint8_t>(uuid.bytes[2]),
                         static_cast<uint8_t>(uuid.bytes[3]),
                         static_cast<uint8_t>(uuid.bytes[4]),
                         static_cast<uint8_t>(uuid.bytes[5]),
                         static_cast<uint8_t>(uuid.bytes[6]),
                         static_cast<uint8_t>(uuid.bytes[7]),
                         static_cast<uint8_t>(uuid.bytes[8]),
                         static_cast<uint8_t>(uuid.bytes[9]),
                         static_cast<uint8_t>(uuid.bytes[10]),
                         static_cast<uint8_t>(uuid.bytes[11]),
                         static_cast<uint8_t>(uuid.bytes[12]),
                         static_cast<uint8_t>(uuid.bytes[13]),
                         static_cast<uint8_t>(uuid.bytes[14]),
                         static_cast<uint8_t>(uuid.bytes[15]));
            dev.uuid = uuid_str;
        }

        info.devices.push_back(std::move(dev));
    }

    SENTINEL_LOG_INFO("Detected {} CUDA devices (driver {}, runtime {})",
                      info.devices.size(), info.driver_version,
                      info.runtime_version);
    return true;
}

}  // namespace

PlatformInfo detect_platform() {
    PlatformInfo info;

    // Try CUDA first.
    if (detect_cuda(info)) {
        return info;
    }

    // HIP detection would go here when SENTINEL_ENABLE_ROCM is set.
    SENTINEL_LOG_WARN("No supported GPU runtime detected");
    info.runtime = GpuRuntime::kNone;
    return info;
}

const char* runtime_to_string(GpuRuntime rt) {
    switch (rt) {
        case GpuRuntime::kCuda: return "CUDA";
        case GpuRuntime::kHip:  return "HIP/ROCm";
        case GpuRuntime::kNone: return "None";
    }
    return "Unknown";
}

}  // namespace sentinel::platform
