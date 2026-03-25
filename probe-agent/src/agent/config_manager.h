/// @file config_manager.h
/// @brief Agent configuration loading, validation, and hot-reload.
///
/// Manages the probe agent's configuration from JSON files, environment
/// variables, and dynamic updates received via gRPC from the Correlation
/// Engine. Supports atomic hot-reload with fallback to last known good.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "probes/probe_interface.h"
#include "util/logging.h"
#include "util/metrics.h"

namespace sentinel::agent {

/// Per-probe-type scheduling configuration.
struct ProbeScheduleConfig {
    probes::ProbeType probe_type;
    uint32_t period_seconds = 60;
    probes::Priority priority = probes::Priority::kHigh;
    probes::SmSelection sm_selection = probes::SmSelection::kAll;
    bool enabled = true;
};

/// gRPC connection configuration.
struct GrpcConfig {
    std::string endpoint = "localhost:50051";
    bool use_tls = false;
    std::string tls_cert_path;
    std::string tls_key_path;
    std::string tls_ca_path;
    std::chrono::seconds connect_timeout{10};
    std::chrono::seconds keepalive_interval{30};
    uint32_t max_retry_attempts = 10;
    std::chrono::seconds initial_backoff{1};
    std::chrono::seconds max_backoff{300};
    uint32_t max_batch_size = 100;
    std::chrono::milliseconds batch_timeout{500};
    std::chrono::hours local_buffer_duration{1};
};

/// Overall agent configuration.
struct AgentConfig {
    /// Agent identity.
    std::string hostname;
    std::string agent_id;

    /// HMAC key for signing probe results.
    std::string hmac_key;

    /// GPU device indices to manage (empty = all detected GPUs).
    std::vector<int> gpu_devices;

    /// Memory budget per GPU in bytes.
    std::size_t gpu_memory_budget = 128 * 1024 * 1024;

    /// Number of executor threads per GPU.
    int executor_threads_per_gpu = 1;

    /// Probe schedule configuration.
    std::vector<ProbeScheduleConfig> probe_schedules;

    /// GPU utilization threshold for probe back-off (percent).
    float utilization_backoff_threshold = 90.0f;

    /// GPU utilization threshold for burst probing (percent).
    float utilization_burst_threshold = 10.0f;

    /// Telemetry polling interval.
    std::chrono::seconds telemetry_interval{1};

    /// Watchdog timeout: if no probe completes within this, log critical.
    std::chrono::seconds watchdog_timeout{120};

    /// gRPC configuration.
    GrpcConfig grpc;

    /// Logging configuration.
    util::LogConfig logging;

    /// Metrics configuration.
    util::MetricsConfig metrics;
};

/// Callback type for configuration change notifications.
using ConfigChangeCallback = std::function<void(const AgentConfig&)>;

/// Manages agent configuration with hot-reload support.
class ConfigManager {
public:
    ConfigManager();
    ~ConfigManager();

    /// Load configuration from a JSON file.
    /// @param path  Path to the JSON configuration file.
    /// @return true if the configuration was loaded and validated.
    bool load_from_file(const std::string& path);

    /// Load configuration from a JSON string.
    bool load_from_string(const std::string& json_str);

    /// Apply default configuration (used when no config file is available).
    void load_defaults();

    /// Get the current configuration (thread-safe snapshot).
    [[nodiscard]] AgentConfig get_config() const;

    /// Apply a probe schedule override (from gRPC ack).
    /// @param probe_type     The probe type to override.
    /// @param period_seconds New period (0 = disable).
    /// @param duration_seconds How long to apply the override.
    void apply_schedule_override(probes::ProbeType probe_type,
                                  uint32_t period_seconds,
                                  uint32_t duration_seconds);

    /// Register a callback for configuration changes.
    void on_config_change(ConfigChangeCallback callback);

    /// Attempt to reload the configuration from the last loaded file.
    /// On failure, the previous configuration is retained.
    bool reload();

    /// @return Path to the currently loaded configuration file.
    [[nodiscard]] std::string config_file_path() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::agent
