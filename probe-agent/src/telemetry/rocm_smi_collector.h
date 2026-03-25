/// @file rocm_smi_collector.h
/// @brief ROCm SMI telemetry collector for AMD GPU monitoring.
///
/// Collects thermal, power, utilization, RAS (ECC), and clock data
/// from AMD GPUs via the ROCm SMI library (rocm_smi_lib).
/// Feature-parity with NvmlCollector for vendor-agnostic telemetry.
/// Compiled only when SENTINEL_ENABLE_ROCM is defined.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sentinel::telemetry {

/// A single AMD GPU telemetry snapshot from ROCm SMI.
/// Mirror of NvmlSnapshot to allow vendor-agnostic telemetry processing.
struct RocmSnapshot {
    int device_index = -1;
    std::string gpu_uuid;

    // Thermal.
    float temperature_edge_c = 0.0f;      ///< Edge (die) temperature.
    float temperature_junction_c = 0.0f;   ///< Junction (hotspot) temperature.
    float temperature_memory_c = 0.0f;     ///< HBM temperature.
    float fan_speed_pct = 0.0f;
    bool throttle_active = false;

    // Power.
    float power_w = 0.0f;
    uint32_t voltage_mv = 0;
    float power_cap_w = 0.0f;
    bool power_throttle_active = false;

    // Clocks.
    uint32_t gpu_clock_mhz = 0;           ///< SCLK (shader/GFX clock).
    uint32_t mem_clock_mhz = 0;           ///< MCLK (memory clock).

    // Utilization.
    float gpu_utilization_pct = 0.0f;
    float memory_utilization_pct = 0.0f;
    uint64_t vram_used_bytes = 0;
    uint64_t vram_total_bytes = 0;

    // RAS (Reliability, Availability, Serviceability) — AMD's ECC equivalent.
    uint64_t correctable_errors = 0;
    uint64_t uncorrectable_errors = 0;
    uint64_t ras_ecc_corrected_sdma = 0;
    uint64_t ras_ecc_uncorrected_sdma = 0;
    uint64_t ras_ecc_corrected_gfx = 0;
    uint64_t ras_ecc_uncorrected_gfx = 0;

    // PCIe.
    uint32_t pcie_gen = 0;
    uint32_t pcie_width = 0;
    uint64_t pcie_bandwidth_sent = 0;      ///< Bytes sent since last query.
    uint64_t pcie_bandwidth_received = 0;  ///< Bytes received since last query.

    // xGMI (inter-GPU fabric) — AMD equivalent of NVLink.
    uint64_t xgmi_read_bytes = 0;
    uint64_t xgmi_write_bytes = 0;
};

/// ROCm SMI telemetry collector.
///
/// Enumerates AMD GPUs via rsmi_init() and collects per-device telemetry
/// at the requested polling rate. Thread-safe for concurrent collect() calls.
class RocmSmiCollector {
public:
    RocmSmiCollector();
    ~RocmSmiCollector();

    RocmSmiCollector(const RocmSmiCollector&) = delete;
    RocmSmiCollector& operator=(const RocmSmiCollector&) = delete;

    /// Initialize ROCm SMI and discover GPU devices.
    /// @return true if ROCm SMI initialized and at least one device found.
    bool initialize();

    /// Shut down ROCm SMI.
    void shutdown();

    /// Collect a snapshot from all AMD GPUs.
    [[nodiscard]] std::vector<RocmSnapshot> collect();

    /// Collect a snapshot from a specific GPU device.
    /// @param device_index ROCm device index.
    /// @return Snapshot for the requested GPU.
    [[nodiscard]] RocmSnapshot collect_device(int device_index);

    /// @return Number of ROCm SMI-managed devices.
    [[nodiscard]] int device_count() const;

    /// @return true if ROCm SMI is initialized and operational.
    [[nodiscard]] bool is_available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::telemetry
