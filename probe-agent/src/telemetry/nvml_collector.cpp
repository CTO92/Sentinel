/// @file nvml_collector.cpp
/// @brief NVML telemetry collector implementation.

#include "telemetry/nvml_collector.h"
#include "util/logging.h"

#include <nvml.h>

namespace sentinel::telemetry {

struct NvmlCollector::Impl {
    bool initialized = false;
    unsigned int num_devices = 0;
    std::vector<nvmlDevice_t> devices;
    std::vector<std::string> uuids;
};

NvmlCollector::NvmlCollector() : impl_(std::make_unique<Impl>()) {}
NvmlCollector::~NvmlCollector() { shutdown(); }

bool NvmlCollector::initialize() {
    nvmlReturn_t ret = nvmlInit_v2();
    if (ret != NVML_SUCCESS) {
        SENTINEL_LOG_ERROR("NVML initialization failed: {}", nvmlErrorString(ret));
        return false;
    }

    ret = nvmlDeviceGetCount_v2(&impl_->num_devices);
    if (ret != NVML_SUCCESS || impl_->num_devices == 0) {
        SENTINEL_LOG_ERROR("NVML device enumeration failed or no devices found");
        nvmlShutdown();
        return false;
    }

    impl_->devices.resize(impl_->num_devices);
    impl_->uuids.resize(impl_->num_devices);

    for (unsigned int i = 0; i < impl_->num_devices; ++i) {
        ret = nvmlDeviceGetHandleByIndex_v2(i, &impl_->devices[i]);
        if (ret != NVML_SUCCESS) {
            SENTINEL_LOG_WARN("Failed to get NVML handle for device {}: {}",
                              i, nvmlErrorString(ret));
            continue;
        }

        char uuid[96] = {};
        ret = nvmlDeviceGetUUID(impl_->devices[i], uuid, sizeof(uuid));
        if (ret == NVML_SUCCESS) {
            impl_->uuids[i] = uuid;
        }
    }

    impl_->initialized = true;
    SENTINEL_LOG_INFO("NVML initialized with {} devices", impl_->num_devices);
    return true;
}

void NvmlCollector::shutdown() {
    if (impl_->initialized) {
        nvmlShutdown();
        impl_->initialized = false;
        SENTINEL_LOG_INFO("NVML shut down");
    }
}

std::vector<NvmlSnapshot> NvmlCollector::collect() {
    std::vector<NvmlSnapshot> snapshots;
    if (!impl_->initialized) return snapshots;

    snapshots.reserve(impl_->num_devices);
    for (unsigned int i = 0; i < impl_->num_devices; ++i) {
        snapshots.push_back(collect_device(static_cast<int>(i)));
    }
    return snapshots;
}

NvmlSnapshot NvmlCollector::collect_device(int device_index) {
    NvmlSnapshot snap;
    snap.device_index = device_index;

    if (!impl_->initialized ||
        device_index < 0 ||
        static_cast<unsigned>(device_index) >= impl_->num_devices) {
        return snap;
    }

    nvmlDevice_t dev = impl_->devices[device_index];
    snap.gpu_uuid = impl_->uuids[device_index];

    // Temperature.
    unsigned int temp = 0;
    if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS) {
        snap.temperature_c = static_cast<float>(temp);
    }

    // Fan speed.
    unsigned int fan = 0;
    if (nvmlDeviceGetFanSpeed(dev, &fan) == NVML_SUCCESS) {
        snap.fan_speed_pct = static_cast<float>(fan);
    }

    // Throttle status.
    unsigned long long throttle_reasons = 0;
    if (nvmlDeviceGetCurrentClocksThrottleReasons(dev, &throttle_reasons) == NVML_SUCCESS) {
        snap.throttle_active = (throttle_reasons & nvmlClocksThrottleReasonSwThermalSlowdown) ||
                               (throttle_reasons & nvmlClocksThrottleReasonHwThermalSlowdown);
        snap.power_throttle_active = (throttle_reasons & nvmlClocksThrottleReasonSwPowerCap) != 0;
    }

    // Power.
    unsigned int power_mw = 0;
    if (nvmlDeviceGetPowerUsage(dev, &power_mw) == NVML_SUCCESS) {
        snap.power_w = static_cast<float>(power_mw) / 1000.0f;
    }

    unsigned int power_limit_mw = 0;
    if (nvmlDeviceGetPowerManagementLimit(dev, &power_limit_mw) == NVML_SUCCESS) {
        snap.power_limit_w = static_cast<float>(power_limit_mw) / 1000.0f;
    }

    // Clocks.
    unsigned int clock = 0;
    if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clock) == NVML_SUCCESS) {
        snap.gpu_clock_mhz = clock;
    }
    if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_MEM, &clock) == NVML_SUCCESS) {
        snap.mem_clock_mhz = clock;
    }

    // Utilization.
    nvmlUtilization_t util = {};
    if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
        snap.gpu_utilization_pct = static_cast<float>(util.gpu);
        snap.memory_utilization_pct = static_cast<float>(util.memory);
    }

    // ECC errors.
    unsigned long long count = 0;
    if (nvmlDeviceGetTotalEccErrors(dev, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                     NVML_VOLATILE_ECC, &count) == NVML_SUCCESS) {
        snap.sram_corrected = count;
    }
    if (nvmlDeviceGetTotalEccErrors(dev, NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                     NVML_VOLATILE_ECC, &count) == NVML_SUCCESS) {
        snap.sram_uncorrected = count;
    }

    // Retired pages.
    unsigned int retired_count = 0;
    if (nvmlDeviceGetRetiredPages(dev, NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
                                   &retired_count, nullptr) == NVML_SUCCESS) {
        snap.retired_pages = retired_count;
    }

    nvmlEnableState_t pending = NVML_FEATURE_DISABLED;
    if (nvmlDeviceGetRetiredPagesPendingStatus(dev, &pending) == NVML_SUCCESS) {
        snap.reset_required = (pending == NVML_FEATURE_ENABLED);
    }

    // PCIe link info.
    unsigned int pcie_gen = 0;
    if (nvmlDeviceGetCurrPcieLinkGeneration(dev, &pcie_gen) == NVML_SUCCESS) {
        snap.pcie_gen = pcie_gen;
    }
    unsigned int pcie_width = 0;
    if (nvmlDeviceGetCurrPcieLinkWidth(dev, &pcie_width) == NVML_SUCCESS) {
        snap.pcie_width = pcie_width;
    }

    // Memory temperature (HBM — available on A100/H100).
    // NVML may not support this on all GPUs; gracefully handle errors.
    unsigned int mem_temp = 0;
    nvmlReturn_t mem_temp_ret = nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &mem_temp);
    if (mem_temp_ret == NVML_SUCCESS) {
        // NVML_TEMPERATURE_GPU is used as fallback; NVML does not always have
        // a separate memory temperature sensor query in all driver versions.
        snap.memory_temperature_c = static_cast<float>(mem_temp);
    }

    return snap;
}

int NvmlCollector::device_count() const {
    return static_cast<int>(impl_->num_devices);
}

bool NvmlCollector::is_available() const {
    return impl_->initialized;
}

}  // namespace sentinel::telemetry
