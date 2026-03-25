/// @file thermal_monitor.cpp
/// @brief GPU thermal monitoring implementation.

#include "telemetry/thermal_monitor.h"
#include "util/logging.h"

#include <deque>
#include <mutex>
#include <unordered_map>

namespace sentinel::telemetry {

struct PerDeviceThermal {
    float max_temp = 0.0f;
    float last_temp = 0.0f;
    bool throttling = false;
    std::deque<std::pair<std::chrono::steady_clock::time_point, float>> history;
    std::chrono::steady_clock::time_point last_update;
};

struct ThermalMonitor::Impl {
    ThermalConfig config;
    mutable std::mutex mutex;
    std::unordered_map<int, PerDeviceThermal> devices;
};

ThermalMonitor::ThermalMonitor(const ThermalConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

ThermalMonitor::~ThermalMonitor() = default;

void ThermalMonitor::update(int device_index, float temp_c, float mem_temp_c,
                             bool throttling) {
    std::lock_guard lock(impl_->mutex);
    auto& dev = impl_->devices[device_index];

    auto now = std::chrono::steady_clock::now();
    dev.last_temp = temp_c;
    dev.throttling = throttling;
    dev.last_update = now;

    if (temp_c > dev.max_temp) {
        dev.max_temp = temp_c;
    }

    // Add to history for rolling average.
    dev.history.emplace_back(now, temp_c);

    // Trim history to the configured window.
    auto cutoff = now - std::chrono::seconds(impl_->config.averaging_window_seconds);
    while (!dev.history.empty() && dev.history.front().first < cutoff) {
        dev.history.pop_front();
    }

    // Log warnings on threshold crossings.
    if (temp_c >= impl_->config.critical_threshold_c) {
        SENTINEL_LOG_CRITICAL("GPU {} temperature CRITICAL: {:.1f}C", device_index, temp_c);
    } else if (temp_c >= impl_->config.warning_threshold_c) {
        SENTINEL_LOG_WARN("GPU {} temperature WARNING: {:.1f}C", device_index, temp_c);
    }
}

ThermalState ThermalMonitor::get_state(int device_index) const {
    std::lock_guard lock(impl_->mutex);
    ThermalState state;
    state.device_index = device_index;

    auto it = impl_->devices.find(device_index);
    if (it == impl_->devices.end()) {
        return state;
    }

    const auto& dev = it->second;
    state.current_temp_c = dev.last_temp;
    state.max_temp_c = dev.max_temp;
    state.is_throttling = dev.throttling;
    state.is_critical = dev.last_temp >= impl_->config.critical_threshold_c;
    state.is_warning = dev.last_temp >= impl_->config.warning_threshold_c;
    state.timestamp = std::chrono::system_clock::now();

    // Compute rolling average.
    if (!dev.history.empty()) {
        float sum = 0.0f;
        for (const auto& [tp, t] : dev.history) {
            sum += t;
        }
        state.avg_temp_c = sum / static_cast<float>(dev.history.size());
    }

    // Compute rate of change (linear regression-ish: delta over last 10s).
    if (dev.history.size() >= 2) {
        auto recent = dev.history.back();
        // Find entry ~10 seconds ago.
        auto target_time = recent.first - std::chrono::seconds(10);
        float old_temp = dev.history.front().second;
        float time_delta_s = 0.0f;
        for (const auto& [tp, t] : dev.history) {
            if (tp <= target_time) {
                old_temp = t;
                auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(
                    recent.first - tp);
                time_delta_s = static_cast<float>(dur.count()) / 1000.0f;
            }
        }
        if (time_delta_s > 0.0f) {
            state.temp_rate_c_per_sec = (recent.second - old_temp) / time_delta_s;
        }
    }

    return state;
}

bool ThermalMonitor::should_backoff(int device_index) const {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->devices.find(device_index);
    if (it == impl_->devices.end()) {
        return false;
    }
    return it->second.last_temp >= impl_->config.probe_backoff_threshold_c ||
           it->second.throttling;
}

void ThermalMonitor::reset(int device_index) {
    std::lock_guard lock(impl_->mutex);
    impl_->devices.erase(device_index);
}

}  // namespace sentinel::telemetry
