/// @file voltage_monitor.h
/// @brief GPU voltage and power monitoring.
///
/// Tracks voltage, current, and power draw trends. Detects anomalous
/// power consumption that may correlate with SDC events (e.g., voltage
/// droops causing logic errors).

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

namespace sentinel::telemetry {

/// Power state assessment for a single GPU.
struct PowerState {
    int device_index = -1;
    float current_power_w = 0.0f;
    float power_limit_w = 0.0f;
    uint32_t voltage_mv = 0;
    float avg_power_w = 0.0f;       ///< Rolling average (60s window).
    float max_power_w = 0.0f;       ///< Maximum observed.
    bool power_throttling = false;
    float headroom_w = 0.0f;        ///< power_limit - current_power.
    std::chrono::system_clock::time_point timestamp;
};

/// Configuration for power monitoring.
struct PowerConfig {
    float high_power_threshold_pct = 90.0f;  ///< % of limit to trigger warning.
    int averaging_window_seconds = 60;
};

/// GPU power/voltage monitor.
class VoltageMonitor {
public:
    explicit VoltageMonitor(const PowerConfig& config = {});
    ~VoltageMonitor();

    /// Update power state from a new reading.
    void update(int device_index, float power_w, float power_limit_w,
                uint32_t voltage_mv, bool throttling);

    /// Get the current power state for a GPU.
    [[nodiscard]] PowerState get_state(int device_index) const;

    /// Check if the GPU is under power stress (close to limit).
    [[nodiscard]] bool is_power_stressed(int device_index) const;

    /// Reset statistics for a GPU.
    void reset(int device_index);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::telemetry
