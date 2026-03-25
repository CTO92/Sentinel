/// @file logging.h
/// @brief Centralized logging initialization and access for the probe agent.
///
/// Wraps spdlog with SENTINEL-specific log formatting, multi-sink support
/// (console + file + optional syslog), and structured key-value fields.

#pragma once

#include <memory>
#include <string>
#include <string_view>

#include <spdlog/spdlog.h>
#include <spdlog/logger.h>

namespace sentinel::util {

/// Logging configuration loaded from the agent config file.
struct LogConfig {
    /// Log level: trace, debug, info, warn, error, critical.
    std::string level = "info";

    /// Path to the log file. Empty disables file logging.
    std::string file_path = "/var/log/sentinel/probe-agent.log";

    /// Maximum size in bytes before log rotation (default 100 MB).
    std::size_t max_file_size = 100 * 1024 * 1024;

    /// Number of rotated log files to keep.
    std::size_t max_files = 5;

    /// Whether to also log to stdout.
    bool console_enabled = true;

    /// Whether to log to syslog (Linux only).
    bool syslog_enabled = false;

    /// Syslog identity string.
    std::string syslog_ident = "sentinel-probe-agent";
};

/// Initialize the global logging system. Must be called once at startup
/// before any log calls. Thread-safe: subsequent calls are no-ops.
void initialize_logging(const LogConfig& config);

/// Shut down the logging system, flushing all sinks. Called at agent teardown.
void shutdown_logging();

/// Retrieve the shared logger instance. Returns spdlog's default logger
/// before initialize_logging() is called.
std::shared_ptr<spdlog::logger> get_logger();

/// Set the runtime log level (e.g., from a config reload).
void set_log_level(std::string_view level);

}  // namespace sentinel::util

// Convenience macros that include source location.
#define SENTINEL_LOG_TRACE(...)    SPDLOG_LOGGER_TRACE(::sentinel::util::get_logger(), __VA_ARGS__)
#define SENTINEL_LOG_DEBUG(...)    SPDLOG_LOGGER_DEBUG(::sentinel::util::get_logger(), __VA_ARGS__)
#define SENTINEL_LOG_INFO(...)     SPDLOG_LOGGER_INFO(::sentinel::util::get_logger(), __VA_ARGS__)
#define SENTINEL_LOG_WARN(...)     SPDLOG_LOGGER_WARN(::sentinel::util::get_logger(), __VA_ARGS__)
#define SENTINEL_LOG_ERROR(...)    SPDLOG_LOGGER_ERROR(::sentinel::util::get_logger(), __VA_ARGS__)
#define SENTINEL_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(::sentinel::util::get_logger(), __VA_ARGS__)
