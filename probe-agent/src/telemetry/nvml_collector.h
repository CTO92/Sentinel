/// @file nvml_collector.h
/// @brief NVIDIA Management Library (NVML) telemetry collector.
///
/// Polls GPU thermal, power, utilization, ECC, and clock data via NVML.
/// Runs on the telemetry thread at 1 Hz and pushes readings into the
/// ring buffer for batching and gRPC transmission.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sentinel::telemetry {

/// A single GPU telemetry snapshot from NVML.
struct NvmlSnapshot {
    int device_index = -1;
    std::string gpu_uuid;

    // Thermal.
    float temperature_c = 0.0f;
    float fan_speed_pct = 0.0f;
    bool throttle_active = false;
    float memory_temperature_c = 0.0f;

    // Power.
    float power_w = 0.0f;
    uint32_t voltage_mv = 0;
    float power_limit_w = 0.0f;
    bool power_throttle_active = false;

    // Clocks.
    uint32_t gpu_clock_mhz = 0;
    uint32_t mem_clock_mhz = 0;

    // Utilization.
    float gpu_utilization_pct = 0.0f;
    float memory_utilization_pct = 0.0f;

    // ECC.
    uint64_t sram_corrected = 0;
    uint64_t sram_uncorrected = 0;
    uint64_t dram_corrected = 0;
    uint64_t dram_uncorrected = 0;
    uint32_t retired_pages = 0;
    uint32_t pending_retired_pages = 0;
    bool reset_required = false;

    // PCIe.
    uint32_t pcie_gen = 0;
    uint32_t pcie_width = 0;
};

/// NVML-based GPU telemetry collector.
class NvmlCollector {
public:
    NvmlCollector();
    ~NvmlCollector();

    NvmlCollector(const NvmlCollector&) = delete;
    NvmlCollector& operator=(const NvmlCollector&) = delete;

    /// Initialize NVML and discover GPU devices.
    /// @return true if NVML initialized and at least one device found.
    bool initialize();

    /// Shut down NVML.
    void shutdown();

    /// Collect a snapshot from all GPUs.
    /// @return Vector of snapshots, one per GPU.
    [[nodiscard]] std::vector<NvmlSnapshot> collect();

    /// Collect a snapshot from a specific GPU device.
    /// @param device_index CUDA device index.
    /// @return Snapshot for the requested GPU.
    [[nodiscard]] NvmlSnapshot collect_device(int device_index);

    /// @return Number of NVML-managed devices.
    [[nodiscard]] int device_count() const;

    /// @return true if NVML is initialized and operational.
    [[nodiscard]] bool is_available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::telemetry
