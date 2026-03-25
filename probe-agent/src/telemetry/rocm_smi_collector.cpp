/// @file rocm_smi_collector.cpp
/// @brief ROCm SMI collector stub implementation.

#include "telemetry/rocm_smi_collector.h"
#include "util/logging.h"

namespace sentinel::telemetry {

struct RocmSmiCollector::Impl {
    bool initialized = false;
};

RocmSmiCollector::RocmSmiCollector() : impl_(std::make_unique<Impl>()) {}
RocmSmiCollector::~RocmSmiCollector() { shutdown(); }

bool RocmSmiCollector::initialize() {
#ifdef SENTINEL_ENABLE_ROCM
    // ROCm SMI initialization would go here:
    //   rsmi_status_t ret = rsmi_init(0);
    //   if (ret != RSMI_STATUS_SUCCESS) { ... }
    SENTINEL_LOG_INFO("ROCm SMI initialization (stub)");
    impl_->initialized = true;
    return true;
#else
    SENTINEL_LOG_DEBUG("ROCm SMI not compiled in; skipping initialization");
    return false;
#endif
}

void RocmSmiCollector::shutdown() {
#ifdef SENTINEL_ENABLE_ROCM
    if (impl_->initialized) {
        // rsmi_shut_down();
        impl_->initialized = false;
    }
#endif
}

std::vector<RocmSnapshot> RocmSmiCollector::collect() {
    std::vector<RocmSnapshot> snapshots;
#ifdef SENTINEL_ENABLE_ROCM
    if (!impl_->initialized) return snapshots;

    // Enumerate devices and collect telemetry via rsmi_* APIs.
    // uint32_t num_devices = 0;
    // rsmi_num_monitor_devices(&num_devices);
    // for (uint32_t i = 0; i < num_devices; ++i) { ... }
#endif
    return snapshots;
}

bool RocmSmiCollector::is_available() const {
    return impl_->initialized;
}

}  // namespace sentinel::telemetry
