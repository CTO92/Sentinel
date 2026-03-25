/// @file metrics_noop.cpp
/// @brief No-op metrics implementation when prometheus-cpp is unavailable.
///
/// Compiled only when SENTINEL_HAS_PROMETHEUS is NOT defined. All methods
/// are intentional no-ops so the rest of the codebase can call metrics
/// unconditionally.

#include "util/metrics.h"

#include <cassert>
#include <mutex>

namespace sentinel::util {

struct Metrics::Impl {};

Metrics* Metrics::s_instance = nullptr;

void Metrics::initialize(const MetricsConfig& /*config*/) {
    static std::once_flag flag;
    std::call_once(flag, []() {
        s_instance = new Metrics();
        s_instance->impl_ = std::make_unique<Impl>();
    });
}

void Metrics::shutdown() {
    if (s_instance) {
        s_instance->impl_.reset();
        delete s_instance;
        s_instance = nullptr;
    }
}

Metrics& Metrics::instance() {
    assert(s_instance && "Metrics::initialize() must be called first");
    return *s_instance;
}

void Metrics::record_probe_execution(std::string_view, std::string_view) {}
void Metrics::record_probe_latency(std::string_view, double) {}
void Metrics::set_active_sm_count(std::string_view, int) {}
void Metrics::record_batch_sent() {}
void Metrics::record_grpc_error(std::string_view) {}
void Metrics::set_grpc_connected(bool) {}
void Metrics::set_buffer_depth(std::size_t) {}
void Metrics::record_telemetry_collection() {}
void Metrics::record_telemetry_error(std::string_view) {}
void Metrics::set_uptime(double) {}
void Metrics::set_gpu_count(int) {}

}  // namespace sentinel::util
