/// @file agent.cpp
/// @brief Probe Agent orchestrator implementation.
///
/// Coordinates the full agent lifecycle:
///   1. Parse configuration (file or defaults).
///   2. Detect GPU platform and enumerate devices.
///   3. Initialize probes on each GPU.
///   4. Start telemetry collection.
///   5. Start the scheduler and executor threads.
///   6. Start gRPC streaming to the Correlation Engine.
///   7. Monitor watchdog and handle signals.

#include "agent/agent.h"
#include "agent/config_manager.h"
#include "agent/grpc_client.h"
#include "agent/scheduler.h"
#include "platform/cuda_runtime.h"
#include "platform/platform_detect.h"
#include "probes/fma_probe.h"
#include "probes/tensor_core_probe.h"
#include "probes/transcendental_probe.h"
#include "probes/aes_probe.h"
#include "probes/memory_probe.h"
#include "probes/register_file_probe.h"
#include "probes/shared_memory_probe.h"
#include "probes/probe_interface.h"
#include "probes/sm_affinity.h"
#include "telemetry/nvml_collector.h"
#include "telemetry/thermal_monitor.h"
#include "telemetry/voltage_monitor.h"
#include "util/crypto.h"
#include "util/logging.h"
#include "util/metrics.h"
#include "util/ring_buffer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace sentinel::agent {

// ── Per-GPU state ───────────────────────────────────────────────────

struct PerGpuState {
    int device_index = -1;
    std::string uuid;
    int sm_count = 0;

    // One probe instance per type, per GPU.
    std::map<probes::ProbeType, std::unique_ptr<probes::ProbeInterface>> probes;

    // CUDA stream for probe execution.
    std::unique_ptr<platform::CudaStream> exec_stream;
};

// ── Agent::Impl ─────────────────────────────────────────────────────

struct Agent::Impl {
    std::atomic<bool> running{false};
    std::atomic<bool> shutdown_requested{false};

    ConfigManager config_manager;
    Scheduler scheduler;
    GrpcClient grpc_client;

    platform::PlatformInfo platform_info;
    std::vector<PerGpuState> gpu_states;

    telemetry::NvmlCollector nvml_collector;
    telemetry::ThermalMonitor thermal_monitor;
    telemetry::VoltageMonitor voltage_monitor;

    // Telemetry thread.
    std::thread telemetry_thread;

    // Watchdog thread.
    std::thread watchdog_thread;
    std::chrono::steady_clock::time_point last_probe_completion;
    std::mutex watchdog_mutex;

    // Shutdown synchronization.
    std::mutex shutdown_mutex;
    std::condition_variable shutdown_cv;

    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

    /// Initialize probes on a single GPU.
    bool init_gpu_probes(PerGpuState& gpu) {
        platform::set_device(gpu.device_index);

        gpu.exec_stream = std::make_unique<platform::CudaStream>();

        auto add_probe = [&](auto probe_ptr) {
            auto type = probe_ptr->type();
            if (probe_ptr->initialize(gpu.device_index)) {
                SENTINEL_LOG_INFO("  {} probe initialized on GPU {} (SM count: {})",
                                  probe_ptr->name(), gpu.device_index, gpu.sm_count);
                gpu.probes[type] = std::move(probe_ptr);
            } else {
                SENTINEL_LOG_WARN("  {} probe failed to initialize on GPU {}",
                                  probe_ptr->name(), gpu.device_index);
            }
        };

        add_probe(std::make_unique<probes::FmaProbe>());
        add_probe(std::make_unique<probes::TensorCoreProbe>());
        add_probe(std::make_unique<probes::TranscendentalProbe>());
        add_probe(std::make_unique<probes::AesProbe>());
        add_probe(std::make_unique<probes::MemoryProbe>());
        add_probe(std::make_unique<probes::RegisterFileProbe>());
        add_probe(std::make_unique<probes::SharedMemoryProbe>());

        return !gpu.probes.empty();
    }

    /// Execute a probe task (called by the scheduler).
    void execute_probe(const ProbeTask& task) {
        // Find the GPU state.
        PerGpuState* gpu = nullptr;
        for (auto& g : gpu_states) {
            if (g.device_index == task.device_index) {
                gpu = &g;
                break;
            }
        }
        if (!gpu) {
            SENTINEL_LOG_ERROR("Unknown device index in probe task: {}",
                               task.device_index);
            return;
        }

        // Find the probe.
        auto it = gpu->probes.find(task.probe_type);
        if (it == gpu->probes.end()) {
            SENTINEL_LOG_WARN("Probe type {} not initialized on GPU {}",
                              probes::probe_type_name(task.probe_type),
                              task.device_index);
            return;
        }

        auto& probe = it->second;

        // Execute with error handling and retry.
        probes::ProbeResult result;
        bool success = false;
        int retries = 0;
        constexpr int max_retries = 1;

        while (retries <= max_retries) {
            try {
                platform::set_device(gpu->device_index);
                result = probe->execute(task.sm_id, gpu->exec_stream->get());

                // Fill in GPU-level fields from telemetry.
                result.gpu_uuid = gpu->uuid;
                auto thermal = thermal_monitor.get_state(gpu->device_index);
                result.gpu_temp_c = thermal.current_temp_c;
                auto power = voltage_monitor.get_state(gpu->device_index);
                result.gpu_power_w = power.current_power_w;

                success = true;
                break;

            } catch (const platform::CudaError& e) {
                SENTINEL_LOG_ERROR("CUDA error during probe execution (attempt {}): {}",
                                    retries + 1, e.what());
                ++retries;

                if (retries > max_retries) {
                    // Report error result.
                    result.probe_type = task.probe_type;
                    result.sm_id = task.sm_id;
                    result.gpu_uuid = gpu->uuid;
                    result.match = false;
                    result.timestamp = std::chrono::system_clock::now();

                    // Check for device hang — try reset.
                    if (e.code() == cudaErrorLaunchTimeout ||
                        e.code() == cudaErrorECCUncorrectable) {
                        SENTINEL_LOG_CRITICAL(
                            "GPU {} may be hung or has ECC errors; "
                            "attempting device reset",
                            gpu->device_index);
                        try {
                            platform::reset_device();
                            // Re-initialize probes after reset.
                            init_gpu_probes(*gpu);
                        } catch (...) {
                            SENTINEL_LOG_CRITICAL(
                                "GPU {} device reset failed; entering degraded mode",
                                gpu->device_index);
                        }
                    }
                }
            }
        }

        // Record metrics.
        auto result_str = result.match ? "pass" : "fail";
        if (!success) result_str = "error";
        util::Metrics::instance().record_probe_execution(
            probes::probe_type_name(task.probe_type), result_str);
        util::Metrics::instance().record_probe_latency(
            probes::probe_type_name(task.probe_type),
            static_cast<double>(result.execution_time_ns) * 1e-9);

        // Send result via gRPC.
        grpc_client.send_probe_result(std::move(result));

        // Update watchdog.
        {
            std::lock_guard lock(watchdog_mutex);
            last_probe_completion = std::chrono::steady_clock::now();
        }
    }

    /// Telemetry collection loop (1 Hz).
    void telemetry_loop() {
        SENTINEL_LOG_INFO("Telemetry thread started");
        auto config = config_manager.get_config();

        while (running.load()) {
            auto cycle_start = std::chrono::steady_clock::now();

            if (nvml_collector.is_available()) {
                try {
                    auto snapshots = nvml_collector.collect();

                    // Update monitors.
                    for (const auto& snap : snapshots) {
                        thermal_monitor.update(snap.device_index,
                                                snap.temperature_c,
                                                snap.memory_temperature_c,
                                                snap.throttle_active);
                        voltage_monitor.update(snap.device_index,
                                                snap.power_w, snap.power_limit_w,
                                                snap.voltage_mv,
                                                snap.power_throttle_active);

                        // Update scheduler with utilization.
                        scheduler.update_utilization(snap.device_index,
                                                      snap.gpu_utilization_pct);
                    }

                    // Send via gRPC.
                    grpc_client.send_telemetry(snapshots);

                    util::Metrics::instance().record_telemetry_collection();

                } catch (const std::exception& e) {
                    SENTINEL_LOG_WARN("Telemetry collection failed: {}", e.what());
                    util::Metrics::instance().record_telemetry_error("nvml");
                }
            }

            // Update uptime metric.
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time);
            util::Metrics::instance().set_uptime(
                static_cast<double>(elapsed.count()));

            // Sleep until next cycle.
            auto cycle_end = std::chrono::steady_clock::now();
            auto cycle_duration = cycle_end - cycle_start;
            auto sleep_time = config.telemetry_interval - cycle_duration;
            if (sleep_time > std::chrono::milliseconds(0)) {
                std::this_thread::sleep_for(sleep_time);
            }
        }

        SENTINEL_LOG_INFO("Telemetry thread stopped");
    }

    /// Watchdog thread: detects stalled probe execution.
    void watchdog_loop() {
        SENTINEL_LOG_INFO("Watchdog thread started");
        auto config = config_manager.get_config();

        while (running.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            if (!running.load()) break;

            std::lock_guard lock(watchdog_mutex);
            auto now = std::chrono::steady_clock::now();
            auto since_last = now - last_probe_completion;

            if (since_last > config.watchdog_timeout) {
                SENTINEL_LOG_CRITICAL(
                    "Watchdog: no probe completed in {}s (threshold: {}s)",
                    std::chrono::duration_cast<std::chrono::seconds>(since_last).count(),
                    config.watchdog_timeout.count());
            }
        }

        SENTINEL_LOG_INFO("Watchdog thread stopped");
    }
};

// ── Public interface ────────────────────────────────────────────────

Agent::Agent() : impl_(std::make_unique<Impl>()) {}
Agent::~Agent() {
    request_shutdown();
    wait_for_shutdown();
}

bool Agent::initialize(const std::string& config_path) {
    // Load configuration.
    if (!config_path.empty()) {
        if (!impl_->config_manager.load_from_file(config_path)) {
            SENTINEL_LOG_WARN("Failed to load config from {}; using defaults",
                              config_path);
            impl_->config_manager.load_defaults();
        }
    } else {
        impl_->config_manager.load_defaults();
    }

    auto config = impl_->config_manager.get_config();

    // Initialize logging.
    util::initialize_logging(config.logging);

    // Initialize metrics.
    util::Metrics::initialize(config.metrics);

    SENTINEL_LOG_INFO("SENTINEL Probe Agent v0.1.0 starting");

    // Detect GPU platform.
    impl_->platform_info = platform::detect_platform();
    if (impl_->platform_info.runtime == platform::GpuRuntime::kNone) {
        SENTINEL_LOG_CRITICAL("No GPU runtime detected; cannot start agent");
        return false;
    }

    SENTINEL_LOG_INFO("Detected {} with {} GPU(s)",
                      platform::runtime_to_string(impl_->platform_info.runtime),
                      impl_->platform_info.devices.size());

    // Determine which GPUs to manage.
    std::vector<int> gpu_indices;
    if (config.gpu_devices.empty()) {
        for (const auto& dev : impl_->platform_info.devices) {
            gpu_indices.push_back(dev.device_index);
        }
    } else {
        gpu_indices = config.gpu_devices;
    }

    util::Metrics::instance().set_gpu_count(static_cast<int>(gpu_indices.size()));

    // Initialize probes on each GPU.
    for (int dev_idx : gpu_indices) {
        // Find device info.
        const platform::GpuDeviceInfo* dev_info = nullptr;
        for (const auto& d : impl_->platform_info.devices) {
            if (d.device_index == dev_idx) {
                dev_info = &d;
                break;
            }
        }
        if (!dev_info) {
            SENTINEL_LOG_WARN("GPU device {} not found; skipping", dev_idx);
            continue;
        }

        PerGpuState gpu;
        gpu.device_index = dev_idx;
        gpu.uuid = dev_info->uuid;
        gpu.sm_count = dev_info->sm_count;

        SENTINEL_LOG_INFO("Initializing probes on GPU {} ({}, {} SMs)",
                          dev_idx, dev_info->name, gpu.sm_count);

        if (impl_->init_gpu_probes(gpu)) {
            impl_->gpu_states.push_back(std::move(gpu));
        } else {
            SENTINEL_LOG_ERROR("No probes initialized on GPU {}; skipping", dev_idx);
        }
    }

    if (impl_->gpu_states.empty()) {
        SENTINEL_LOG_CRITICAL("No GPUs with initialized probes; cannot start agent");
        return false;
    }

    // Initialize NVML telemetry.
    if (!impl_->nvml_collector.initialize()) {
        SENTINEL_LOG_WARN("NVML initialization failed; continuing without telemetry");
    }

    // Initialize gRPC client.
    if (!impl_->grpc_client.initialize(config.grpc, config.hostname, config.hmac_key)) {
        SENTINEL_LOG_WARN("gRPC client initialization failed; will retry on start");
    }

    // Register schedule override callback.
    impl_->grpc_client.on_schedule_override(
        [this](probes::ProbeType type, uint32_t period, uint32_t duration) {
            impl_->scheduler.apply_override(type, period, duration);
            impl_->config_manager.apply_schedule_override(type, period, duration);
        });

    // Initialize scheduler.
    std::vector<std::pair<int, int>> device_sm_counts;
    for (const auto& g : impl_->gpu_states) {
        device_sm_counts.emplace_back(g.device_index, g.sm_count);
    }
    impl_->scheduler.initialize(config.probe_schedules, device_sm_counts);

    SENTINEL_LOG_INFO("Agent initialization complete");
    return true;
}

void Agent::run() {
    impl_->running.store(true);
    impl_->last_probe_completion = std::chrono::steady_clock::now();

    // Start gRPC client.
    impl_->grpc_client.start();

    // Start telemetry thread.
    impl_->telemetry_thread = std::thread(&Impl::telemetry_loop, impl_.get());

    // Start watchdog thread.
    impl_->watchdog_thread = std::thread(&Impl::watchdog_loop, impl_.get());

    // Start scheduler (which dispatches to execute_probe).
    impl_->scheduler.start([this](const ProbeTask& task) {
        impl_->execute_probe(task);
    });

    SENTINEL_LOG_INFO("Agent is running");

    // Wait for shutdown signal.
    {
        std::unique_lock lock(impl_->shutdown_mutex);
        impl_->shutdown_cv.wait(lock, [this] {
            return impl_->shutdown_requested.load();
        });
    }

    // Shut down in reverse order.
    SENTINEL_LOG_INFO("Agent shutting down...");

    impl_->scheduler.stop();

    impl_->running.store(false);

    if (impl_->telemetry_thread.joinable()) {
        impl_->telemetry_thread.join();
    }
    if (impl_->watchdog_thread.joinable()) {
        impl_->watchdog_thread.join();
    }

    impl_->grpc_client.stop();

    // Tear down probes.
    for (auto& gpu : impl_->gpu_states) {
        for (auto& [type, probe] : gpu.probes) {
            probe->teardown();
        }
    }

    impl_->nvml_collector.shutdown();

    util::Metrics::shutdown();
    util::shutdown_logging();

    SENTINEL_LOG_INFO("Agent shut down complete");
}

void Agent::request_shutdown() {
    impl_->shutdown_requested.store(true);
    impl_->shutdown_cv.notify_all();
}

void Agent::wait_for_shutdown() {
    // If run() hasn't been called, nothing to wait for.
    if (!impl_->running.load() && !impl_->shutdown_requested.load()) {
        return;
    }
    // The run() method will return when shutdown is complete.
}

bool Agent::is_running() const {
    return impl_->running.load();
}

}  // namespace sentinel::agent
