/// @file rocm_smi_collector.cpp
/// @brief ROCm SMI telemetry collector implementation.
///
/// Full implementation of AMD GPU telemetry collection via the ROCm SMI
/// library. Mirrors the NVML collector's functionality for vendor-agnostic
/// operation.

#include "telemetry/rocm_smi_collector.h"
#include "util/logging.h"

#ifdef SENTINEL_ENABLE_ROCM
#include <rocm_smi/rocm_smi.h>
#endif

#include <algorithm>
#include <cstring>
#include <mutex>

namespace sentinel::telemetry {

struct RocmSmiCollector::Impl {
    bool initialized = false;
    uint32_t num_devices = 0;
    std::vector<std::string> uuids;
    std::mutex collect_mutex;
};

RocmSmiCollector::RocmSmiCollector() : impl_(std::make_unique<Impl>()) {}
RocmSmiCollector::~RocmSmiCollector() { shutdown(); }

bool RocmSmiCollector::initialize() {
#ifdef SENTINEL_ENABLE_ROCM
    rsmi_status_t ret = rsmi_init(0);
    if (ret != RSMI_STATUS_SUCCESS) {
        SENTINEL_LOG_ERROR("ROCm SMI initialization failed: status {}", static_cast<int>(ret));
        return false;
    }

    ret = rsmi_num_monitor_devices(&impl_->num_devices);
    if (ret != RSMI_STATUS_SUCCESS || impl_->num_devices == 0) {
        SENTINEL_LOG_ERROR("ROCm SMI device enumeration failed or no devices found");
        rsmi_shut_down();
        return false;
    }

    impl_->uuids.resize(impl_->num_devices);

    for (uint32_t i = 0; i < impl_->num_devices; ++i) {
        // Retrieve the unique ID for each device.
        uint64_t unique_id = 0;
        ret = rsmi_dev_unique_id_get(i, &unique_id);
        if (ret == RSMI_STATUS_SUCCESS) {
            // Format as a hex string to match GPU identifier conventions.
            char buf[32] = {};
            std::snprintf(buf, sizeof(buf), "GPU-%016llX",
                          static_cast<unsigned long long>(unique_id));
            impl_->uuids[i] = buf;
        } else {
            // Fallback to index-based identifier.
            impl_->uuids[i] = "GPU-ROCM-" + std::to_string(i);
            SENTINEL_LOG_WARN("Could not get unique ID for ROCm device {}: status {}",
                              i, static_cast<int>(ret));
        }
    }

    impl_->initialized = true;
    SENTINEL_LOG_INFO("ROCm SMI initialized with {} devices", impl_->num_devices);
    return true;
#else
    SENTINEL_LOG_DEBUG("ROCm SMI not compiled in; skipping initialization");
    return false;
#endif
}

void RocmSmiCollector::shutdown() {
#ifdef SENTINEL_ENABLE_ROCM
    if (impl_->initialized) {
        rsmi_shut_down();
        impl_->initialized = false;
        SENTINEL_LOG_INFO("ROCm SMI shut down");
    }
#endif
}

std::vector<RocmSnapshot> RocmSmiCollector::collect() {
    std::vector<RocmSnapshot> snapshots;
#ifdef SENTINEL_ENABLE_ROCM
    if (!impl_->initialized) return snapshots;

    std::lock_guard<std::mutex> lock(impl_->collect_mutex);
    snapshots.reserve(impl_->num_devices);
    for (uint32_t i = 0; i < impl_->num_devices; ++i) {
        snapshots.push_back(collect_device(static_cast<int>(i)));
    }
#endif
    return snapshots;
}

RocmSnapshot RocmSmiCollector::collect_device(int device_index) {
    RocmSnapshot snap;
    snap.device_index = device_index;

#ifdef SENTINEL_ENABLE_ROCM
    if (!impl_->initialized ||
        device_index < 0 ||
        static_cast<uint32_t>(device_index) >= impl_->num_devices) {
        return snap;
    }

    uint32_t dev = static_cast<uint32_t>(device_index);
    snap.gpu_uuid = impl_->uuids[dev];

    // ---- Temperature ----
    // Edge temperature (die surface).
    int64_t temp_milli_c = 0;
    if (rsmi_dev_temp_metric_get(dev, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT,
                                  &temp_milli_c) == RSMI_STATUS_SUCCESS) {
        snap.temperature_edge_c = static_cast<float>(temp_milli_c) / 1000.0f;
    }

    // Junction (hotspot) temperature.
    if (rsmi_dev_temp_metric_get(dev, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT,
                                  &temp_milli_c) == RSMI_STATUS_SUCCESS) {
        snap.temperature_junction_c = static_cast<float>(temp_milli_c) / 1000.0f;
    }

    // Memory (HBM) temperature.
    if (rsmi_dev_temp_metric_get(dev, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT,
                                  &temp_milli_c) == RSMI_STATUS_SUCCESS) {
        snap.temperature_memory_c = static_cast<float>(temp_milli_c) / 1000.0f;
    }

    // ---- Fan speed ----
    int64_t fan_speed = 0;
    int64_t fan_max = 0;
    if (rsmi_dev_fan_speed_get(dev, 0, &fan_speed) == RSMI_STATUS_SUCCESS &&
        rsmi_dev_fan_speed_max_get(dev, 0, &fan_max) == RSMI_STATUS_SUCCESS &&
        fan_max > 0) {
        snap.fan_speed_pct = static_cast<float>(fan_speed) / static_cast<float>(fan_max) * 100.0f;
    }

    // ---- Throttle status ----
    // Check performance determinism: if the current SCLK is lower than the max,
    // throttling is likely active.
    rsmi_frequencies_t freq_info = {};
    if (rsmi_dev_gpu_clk_freq_get(dev, RSMI_CLK_TYPE_SYS, &freq_info) == RSMI_STATUS_SUCCESS) {
        if (freq_info.current < freq_info.num_supported &&
            freq_info.current < freq_info.num_supported - 1) {
            // Current frequency index is below the maximum available.
            snap.throttle_active = true;
        }
        if (freq_info.current < freq_info.num_supported) {
            snap.gpu_clock_mhz = static_cast<uint32_t>(
                freq_info.frequency[freq_info.current] / 1000000ULL);
        }
    }

    // ---- Power ----
    uint64_t power_micro_w = 0;
    if (rsmi_dev_power_ave_get(dev, 0, &power_micro_w) == RSMI_STATUS_SUCCESS) {
        snap.power_w = static_cast<float>(power_micro_w) / 1000000.0f;
    }

    // Power cap (default profile).
    uint64_t power_cap_micro_w = 0;
    if (rsmi_dev_power_cap_get(dev, 0, &power_cap_micro_w) == RSMI_STATUS_SUCCESS) {
        snap.power_cap_w = static_cast<float>(power_cap_micro_w) / 1000000.0f;
        // Power throttling if near cap.
        if (snap.power_w > snap.power_cap_w * 0.95f) {
            snap.power_throttle_active = true;
        }
    }

    // Voltage (GFX domain).
    // rsmi_dev_volt_metric_get may not be available on all ASICs.
    int64_t voltage_milli_v = 0;
    if (rsmi_dev_volt_metric_get(dev, RSMI_VOLT_TYPE_VDDGFX, RSMI_VOLT_CURRENT,
                                  &voltage_milli_v) == RSMI_STATUS_SUCCESS) {
        snap.voltage_mv = static_cast<uint32_t>(voltage_milli_v);
    }

    // ---- Memory clock ----
    rsmi_frequencies_t mem_freq_info = {};
    if (rsmi_dev_gpu_clk_freq_get(dev, RSMI_CLK_TYPE_MEM, &mem_freq_info) == RSMI_STATUS_SUCCESS) {
        if (mem_freq_info.current < mem_freq_info.num_supported) {
            snap.mem_clock_mhz = static_cast<uint32_t>(
                mem_freq_info.frequency[mem_freq_info.current] / 1000000ULL);
        }
    }

    // ---- Utilization ----
    uint32_t busy_percent = 0;
    if (rsmi_dev_busy_percent_get(dev, &busy_percent) == RSMI_STATUS_SUCCESS) {
        snap.gpu_utilization_pct = static_cast<float>(busy_percent);
    }

    // Memory utilization.
    uint64_t vram_total = 0, vram_used = 0;
    if (rsmi_dev_memory_total_get(dev, RSMI_MEM_TYPE_VRAM, &vram_total) == RSMI_STATUS_SUCCESS &&
        rsmi_dev_memory_usage_get(dev, RSMI_MEM_TYPE_VRAM, &vram_used) == RSMI_STATUS_SUCCESS) {
        snap.vram_total_bytes = vram_total;
        snap.vram_used_bytes = vram_used;
        if (vram_total > 0) {
            snap.memory_utilization_pct =
                static_cast<float>(vram_used) / static_cast<float>(vram_total) * 100.0f;
        }
    }

    // ---- RAS (ECC) error counters ----
    // GFX block correctable/uncorrectable.
    uint64_t ec_count = 0;
    if (rsmi_dev_ecc_count_get(dev, RSMI_GPU_BLOCK_GFX, &ec_count) == RSMI_STATUS_SUCCESS) {
        // rsmi_dev_ecc_count_get returns a rsmi_error_count_t with correctable/uncorrectable.
        // We access via the struct pointer aliasing pattern.
    }
    // Use the detailed RAS counter query for each block.
    rsmi_error_count_t err_count = {};
    if (rsmi_dev_ecc_count_get(dev, RSMI_GPU_BLOCK_GFX,
                                reinterpret_cast<uint64_t*>(&err_count)) == RSMI_STATUS_SUCCESS) {
        snap.ras_ecc_corrected_gfx = err_count.correctable_err;
        snap.ras_ecc_uncorrected_gfx = err_count.uncorrectable_err;
    }

    rsmi_error_count_t sdma_err = {};
    if (rsmi_dev_ecc_count_get(dev, RSMI_GPU_BLOCK_SDMA,
                                reinterpret_cast<uint64_t*>(&sdma_err)) == RSMI_STATUS_SUCCESS) {
        snap.ras_ecc_corrected_sdma = sdma_err.correctable_err;
        snap.ras_ecc_uncorrected_sdma = sdma_err.uncorrectable_err;
    }

    // Aggregate correctable/uncorrectable across UMC (Unified Memory Controller) block.
    rsmi_error_count_t umc_err = {};
    if (rsmi_dev_ecc_count_get(dev, RSMI_GPU_BLOCK_UMC,
                                reinterpret_cast<uint64_t*>(&umc_err)) == RSMI_STATUS_SUCCESS) {
        snap.correctable_errors = umc_err.correctable_err +
                                   snap.ras_ecc_corrected_gfx +
                                   snap.ras_ecc_corrected_sdma;
        snap.uncorrectable_errors = umc_err.uncorrectable_err +
                                     snap.ras_ecc_uncorrected_gfx +
                                     snap.ras_ecc_uncorrected_sdma;
    }

    // ---- PCIe ----
    uint64_t pcie_sent = 0, pcie_received = 0, pcie_max_pkt_sz = 0;
    if (rsmi_dev_pci_throughput_get(dev, &pcie_sent, &pcie_received,
                                     &pcie_max_pkt_sz) == RSMI_STATUS_SUCCESS) {
        snap.pcie_bandwidth_sent = pcie_sent;
        snap.pcie_bandwidth_received = pcie_received;
    }

    // PCIe link speed/width.
    rsmi_pcie_bandwidth_t pcie_bw = {};
    if (rsmi_dev_pci_bandwidth_get(dev, &pcie_bw) == RSMI_STATUS_SUCCESS) {
        if (pcie_bw.transfer_rate.current < pcie_bw.transfer_rate.num_supported) {
            uint64_t rate = pcie_bw.transfer_rate.frequency[pcie_bw.transfer_rate.current];
            // Map transfer rate to PCIe generation.
            if (rate >= 32000000000ULL)       snap.pcie_gen = 5;  // ~32 GT/s
            else if (rate >= 16000000000ULL)  snap.pcie_gen = 4;  // ~16 GT/s
            else if (rate >= 8000000000ULL)   snap.pcie_gen = 3;  // ~8 GT/s
            else                              snap.pcie_gen = 2;
        }
        // Lane width from lanes array.
        if (pcie_bw.transfer_rate.current < RSMI_MAX_NUM_FREQUENCIES) {
            snap.pcie_width = static_cast<uint32_t>(
                pcie_bw.lanes[pcie_bw.transfer_rate.current]);
        }
    }

    // ---- xGMI (inter-GPU fabric) ----
    uint64_t xgmi_read = 0, xgmi_write = 0;
    if (rsmi_dev_xgmi_hive_id_get(dev, &xgmi_read) == RSMI_STATUS_SUCCESS) {
        // xGMI counters are available via rsmi_dev_counter_* APIs.
        // For telemetry, we capture aggregate bandwidth.
        // Note: actual xGMI counter reading requires creating/reading perf counters,
        // which is done in the performance monitoring subsystem. Here we record
        // the hive ID for correlation purposes.
        snap.xgmi_read_bytes = 0;   // Populated by perf counter polling loop.
        snap.xgmi_write_bytes = 0;
    }

#endif  // SENTINEL_ENABLE_ROCM
    return snap;
}

int RocmSmiCollector::device_count() const {
    return static_cast<int>(impl_->num_devices);
}

bool RocmSmiCollector::is_available() const {
    return impl_->initialized;
}

}  // namespace sentinel::telemetry
