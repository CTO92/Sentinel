/// @file thermal_monitor.h
/// @brief GPU thermal monitoring and throttle detection.
///
/// Wraps NvmlCollector thermal readings with threshold-based alerting,
/// trend analysis, and throttle state tracking. Used by the scheduler
/// to back off probe execution when GPUs are thermally stressed.

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace sentinel::telemetry {

/// Thermal state assessment for a single GPU.
struct ThermalState {
    int device_index = -1;
    float current_temp_c = 0.0f;
    float memory_temp_c = 0.0f;
    float max_temp_c = 0.0f;         ///< Maximum observed since last reset.
    float avg_temp_c = 0.0f;         ///< Rolling average (60s window).
    bool is_throttling = false;
    bool is_critical = false;         ///< Above critical threshold (95C default).
    bool is_warning = false;          ///< Above warning threshold (85C default).
    float temp_rate_c_per_sec = 0.0f; ///< Temperature rate of change.
    std::chrono::system_clock::time_point timestamp;
};

/// Configuration for thermal monitoring thresholds.
struct ThermalConfig {
    float warning_threshold_c = 85.0f;
    float critical_threshold_c = 95.0f;
    float probe_backoff_threshold_c = 80.0f;
    int averaging_window_seconds = 60;
};

/// GPU thermal monitor.
class ThermalMonitor {
public:
    explicit ThermalMonitor(const ThermalConfig& config = {});
    ~ThermalMonitor();

    ThermalMonitor(const ThermalMonitor&) = delete;
    ThermalMonitor& operator=(const ThermalMonitor&) = delete;

    /// Update thermal state from a new temperature reading.
    /// @param device_index GPU device index.
    /// @param temp_c       Current GPU temperature in Celsius.
    /// @param mem_temp_c   Current memory temperature in Celsius.
    /// @param throttling   Whether thermal throttling is active.
    void update(int device_index, float temp_c, float mem_temp_c, bool throttling);

    /// Get the current thermal state for a GPU.
    [[nodiscard]] ThermalState get_state(int device_index) const;

    /// Check if probes should back off on this GPU due to thermal stress.
    [[nodiscard]] bool should_backoff(int device_index) const;

    /// Reset statistics for a GPU (e.g., after a device reset).
    void reset(int device_index);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::telemetry
