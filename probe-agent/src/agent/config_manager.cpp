/// @file config_manager.cpp
/// @brief Agent configuration management implementation.

#include "agent/config_manager.h"
#include "util/logging.h"

#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace sentinel::agent {

using json = nlohmann::json;

namespace {

probes::ProbeType parse_probe_type(const std::string& s) {
    if (s == "fma")            return probes::ProbeType::kFma;
    if (s == "tensor_core")    return probes::ProbeType::kTensorCore;
    if (s == "transcendental") return probes::ProbeType::kTranscendental;
    if (s == "aes")            return probes::ProbeType::kAes;
    if (s == "memory")         return probes::ProbeType::kMemory;
    if (s == "register_file")  return probes::ProbeType::kRegisterFile;
    if (s == "shared_memory")  return probes::ProbeType::kSharedMemory;
    throw std::runtime_error("Unknown probe type: " + s);
}

probes::Priority parse_priority(const std::string& s) {
    if (s == "low")    return probes::Priority::kLow;
    if (s == "medium") return probes::Priority::kMedium;
    if (s == "high")   return probes::Priority::kHigh;
    return probes::Priority::kMedium;
}

probes::SmSelection parse_sm_selection(const std::string& s) {
    if (s == "all")          return probes::SmSelection::kAll;
    if (s == "sample_25pct") return probes::SmSelection::kSample25Pct;
    if (s == "sample_10pct") return probes::SmSelection::kSample10Pct;
    return probes::SmSelection::kAll;
}

AgentConfig parse_config(const json& j) {
    AgentConfig config;

    // Agent identity.
    if (j.contains("hostname")) {
        config.hostname = j["hostname"].get<std::string>();
    } else {
        char hostname_buf[256] = {};
        gethostname(hostname_buf, sizeof(hostname_buf));
        config.hostname = hostname_buf;
    }
    config.agent_id = j.value("agent_id", config.hostname);
    config.hmac_key = j.value("hmac_key", "sentinel-default-key-change-me");

    // GPU devices.
    if (j.contains("gpu_devices")) {
        config.gpu_devices = j["gpu_devices"].get<std::vector<int>>();
    }

    config.gpu_memory_budget = j.value("gpu_memory_budget_mb", 128) * 1024 * 1024;
    config.executor_threads_per_gpu = j.value("executor_threads_per_gpu", 1);
    config.utilization_backoff_threshold = j.value("utilization_backoff_threshold", 90.0f);
    config.utilization_burst_threshold = j.value("utilization_burst_threshold", 10.0f);
    config.telemetry_interval = std::chrono::seconds(j.value("telemetry_interval_seconds", 1));
    config.watchdog_timeout = std::chrono::seconds(j.value("watchdog_timeout_seconds", 120));

    // Probe schedules.
    if (j.contains("probe_schedules")) {
        for (const auto& ps : j["probe_schedules"]) {
            ProbeScheduleConfig psc;
            psc.probe_type = parse_probe_type(ps["type"].get<std::string>());
            psc.period_seconds = ps.value("period_seconds", 60u);
            psc.priority = parse_priority(ps.value("priority", "medium"));
            psc.sm_selection = parse_sm_selection(ps.value("sm_selection", "all"));
            psc.enabled = ps.value("enabled", true);
            config.probe_schedules.push_back(psc);
        }
    }

    // gRPC config.
    if (j.contains("grpc")) {
        const auto& g = j["grpc"];
        config.grpc.endpoint = g.value("endpoint", "localhost:50051");
        config.grpc.use_tls = g.value("use_tls", false);
        config.grpc.tls_cert_path = g.value("tls_cert_path", "");
        config.grpc.tls_key_path = g.value("tls_key_path", "");
        config.grpc.tls_ca_path = g.value("tls_ca_path", "");
        config.grpc.connect_timeout = std::chrono::seconds(g.value("connect_timeout_seconds", 10));
        config.grpc.keepalive_interval = std::chrono::seconds(g.value("keepalive_interval_seconds", 30));
        config.grpc.max_retry_attempts = g.value("max_retry_attempts", 10u);
        config.grpc.max_batch_size = g.value("max_batch_size", 100u);
        config.grpc.batch_timeout = std::chrono::milliseconds(g.value("batch_timeout_ms", 500));
    }

    // Logging config.
    if (j.contains("logging")) {
        const auto& l = j["logging"];
        config.logging.level = l.value("level", "info");
        config.logging.file_path = l.value("file_path", "/var/log/sentinel/probe-agent.log");
        config.logging.console_enabled = l.value("console_enabled", true);
        config.logging.syslog_enabled = l.value("syslog_enabled", false);
    }

    // Metrics config.
    if (j.contains("metrics")) {
        const auto& m = j["metrics"];
        config.metrics.port = m.value("port", static_cast<uint16_t>(9101));
        config.metrics.bind_address = m.value("bind_address", "0.0.0.0");
        config.metrics.enabled = m.value("enabled", true);
    }

    return config;
}

/// Fill in default probe schedules if none were specified.
void ensure_default_schedules(AgentConfig& config) {
    if (!config.probe_schedules.empty()) return;

    config.probe_schedules = {
        {probes::ProbeType::kFma,           60,  probes::Priority::kHigh,   probes::SmSelection::kAll,          true},
        {probes::ProbeType::kTensorCore,    60,  probes::Priority::kHigh,   probes::SmSelection::kAll,          true},
        {probes::ProbeType::kTranscendental,120, probes::Priority::kMedium, probes::SmSelection::kAll,          true},
        {probes::ProbeType::kAes,           300, probes::Priority::kMedium, probes::SmSelection::kSample25Pct,  true},
        {probes::ProbeType::kMemory,        600, probes::Priority::kLow,    probes::SmSelection::kSample10Pct,  true},
        {probes::ProbeType::kRegisterFile,  120, probes::Priority::kMedium, probes::SmSelection::kAll,          true},
        {probes::ProbeType::kSharedMemory,  120, probes::Priority::kMedium, probes::SmSelection::kAll,          true},
    };
}

}  // namespace

struct ConfigManager::Impl {
    mutable std::shared_mutex mutex;
    AgentConfig config;
    std::string config_path;
    std::vector<ConfigChangeCallback> callbacks;

    void notify_change() {
        for (auto& cb : callbacks) {
            try {
                cb(config);
            } catch (const std::exception& e) {
                SENTINEL_LOG_ERROR("Config change callback threw: {}", e.what());
            }
        }
    }
};

ConfigManager::ConfigManager() : impl_(std::make_unique<Impl>()) {}
ConfigManager::~ConfigManager() = default;

bool ConfigManager::load_from_file(const std::string& path) {
    try {
        std::ifstream ifs(path);
        if (!ifs.is_open()) {
            SENTINEL_LOG_ERROR("Cannot open config file: {}", path);
            return false;
        }

        json j = json::parse(ifs);
        AgentConfig new_config = parse_config(j);
        ensure_default_schedules(new_config);

        {
            std::unique_lock lock(impl_->mutex);
            impl_->config = std::move(new_config);
            impl_->config_path = path;
        }

        impl_->notify_change();
        SENTINEL_LOG_INFO("Configuration loaded from {}", path);
        return true;

    } catch (const json::exception& e) {
        SENTINEL_LOG_ERROR("JSON parse error in config: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        SENTINEL_LOG_ERROR("Config load error: {}", e.what());
        return false;
    }
}

bool ConfigManager::load_from_string(const std::string& json_str) {
    try {
        json j = json::parse(json_str);
        AgentConfig new_config = parse_config(j);
        ensure_default_schedules(new_config);

        {
            std::unique_lock lock(impl_->mutex);
            impl_->config = std::move(new_config);
        }

        impl_->notify_change();
        return true;

    } catch (const std::exception& e) {
        SENTINEL_LOG_ERROR("Config parse error: {}", e.what());
        return false;
    }
}

void ConfigManager::load_defaults() {
    AgentConfig config;
    char hostname_buf[256] = {};
    gethostname(hostname_buf, sizeof(hostname_buf));
    config.hostname = hostname_buf;
    config.agent_id = config.hostname;
    ensure_default_schedules(config);

    {
        std::unique_lock lock(impl_->mutex);
        impl_->config = std::move(config);
    }

    impl_->notify_change();
    SENTINEL_LOG_INFO("Default configuration loaded");
}

AgentConfig ConfigManager::get_config() const {
    std::shared_lock lock(impl_->mutex);
    return impl_->config;
}

void ConfigManager::apply_schedule_override(probes::ProbeType probe_type,
                                             uint32_t period_seconds,
                                             uint32_t duration_seconds) {
    std::unique_lock lock(impl_->mutex);
    for (auto& ps : impl_->config.probe_schedules) {
        if (ps.probe_type == probe_type) {
            SENTINEL_LOG_INFO("Applying schedule override for {}: period={}s, duration={}s",
                              probes::probe_type_name(probe_type),
                              period_seconds, duration_seconds);
            ps.period_seconds = period_seconds;
            ps.enabled = (period_seconds > 0);
            // Duration tracking is handled by the scheduler, not the config manager.
            break;
        }
    }
}

void ConfigManager::on_config_change(ConfigChangeCallback callback) {
    impl_->callbacks.push_back(std::move(callback));
}

bool ConfigManager::reload() {
    if (impl_->config_path.empty()) {
        SENTINEL_LOG_WARN("No config file path set; cannot reload");
        return false;
    }
    return load_from_file(impl_->config_path);
}

std::string ConfigManager::config_file_path() const {
    return impl_->config_path;
}

}  // namespace sentinel::agent
