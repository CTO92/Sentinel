/// @file voltage_monitor.cpp
/// @brief GPU voltage and power monitoring implementation.

#include "telemetry/voltage_monitor.h"
#include "util/logging.h"

#include <deque>
#include <mutex>
#include <unordered_map>

namespace sentinel::telemetry {

struct PerDevicePower {
    float max_power = 0.0f;
    float last_power = 0.0f;
    float power_limit = 0.0f;
    uint32_t voltage_mv = 0;
    bool throttling = false;
    std::deque<std::pair<std::chrono::steady_clock::time_point, float>> history;
};

struct VoltageMonitor::Impl {
    PowerConfig config;
    mutable std::mutex mutex;
    std::unordered_map<int, PerDevicePower> devices;
};

VoltageMonitor::VoltageMonitor(const PowerConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

VoltageMonitor::~VoltageMonitor() = default;

void VoltageMonitor::update(int device_index, float power_w, float power_limit_w,
                             uint32_t voltage_mv, bool throttling) {
    std::lock_guard lock(impl_->mutex);
    auto& dev = impl_->devices[device_index];
    auto now = std::chrono::steady_clock::now();

    dev.last_power = power_w;
    dev.power_limit = power_limit_w;
    dev.voltage_mv = voltage_mv;
    dev.throttling = throttling;

    if (power_w > dev.max_power) {
        dev.max_power = power_w;
    }

    dev.history.emplace_back(now, power_w);

    auto cutoff = now - std::chrono::seconds(impl_->config.averaging_window_seconds);
    while (!dev.history.empty() && dev.history.front().first < cutoff) {
        dev.history.pop_front();
    }
}

PowerState VoltageMonitor::get_state(int device_index) const {
    std::lock_guard lock(impl_->mutex);
    PowerState state;
    state.device_index = device_index;

    auto it = impl_->devices.find(device_index);
    if (it == impl_->devices.end()) return state;

    const auto& dev = it->second;
    state.current_power_w = dev.last_power;
    state.power_limit_w = dev.power_limit;
    state.voltage_mv = dev.voltage_mv;
    state.max_power_w = dev.max_power;
    state.power_throttling = dev.throttling;
    state.headroom_w = dev.power_limit - dev.last_power;
    state.timestamp = std::chrono::system_clock::now();

    if (!dev.history.empty()) {
        float sum = 0.0f;
        for (const auto& [tp, p] : dev.history) {
            sum += p;
        }
        state.avg_power_w = sum / static_cast<float>(dev.history.size());
    }

    return state;
}

bool VoltageMonitor::is_power_stressed(int device_index) const {
    std::lock_guard lock(impl_->mutex);
    auto it = impl_->devices.find(device_index);
    if (it == impl_->devices.end()) return false;

    const auto& dev = it->second;
    if (dev.power_limit <= 0.0f) return false;

    float pct = (dev.last_power / dev.power_limit) * 100.0f;
    return pct >= impl_->config.high_power_threshold_pct || dev.throttling;
}

void VoltageMonitor::reset(int device_index) {
    std::lock_guard lock(impl_->mutex);
    impl_->devices.erase(device_index);
}

}  // namespace sentinel::telemetry
