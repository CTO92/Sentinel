/// @file rocm_smi_collector.h
/// @brief ROCm SMI telemetry collector stub.
///
/// Placeholder for AMD GPU telemetry collection via rocm_smi_lib.
/// Compiled only when SENTINEL_ENABLE_ROCM is defined.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sentinel::telemetry {

/// Telemetry snapshot for an AMD GPU (mirrors NvmlSnapshot structure).
struct RocmSnapshot {
    int device_index = -1;
    std::string gpu_uuid;

    float temperature_c = 0.0f;
    float fan_speed_pct = 0.0f;
    float power_w = 0.0f;
    uint32_t gpu_clock_mhz = 0;
    uint32_t mem_clock_mhz = 0;
    float gpu_utilization_pct = 0.0f;
    float memory_utilization_pct = 0.0f;
};

/// ROCm SMI telemetry collector.
class RocmSmiCollector {
public:
    RocmSmiCollector();
    ~RocmSmiCollector();

    /// Initialize ROCm SMI. Returns false if not available.
    bool initialize();

    /// Shut down ROCm SMI.
    void shutdown();

    /// Collect telemetry from all AMD GPUs.
    [[nodiscard]] std::vector<RocmSnapshot> collect();

    /// @return true if ROCm SMI is initialized.
    [[nodiscard]] bool is_available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::telemetry
