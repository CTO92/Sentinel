/// @file logging.cpp
/// @brief Implementation of centralized logging initialization.

#include "util/logging.h"

#include <mutex>

#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#ifdef __linux__
#include <spdlog/sinks/syslog_sink.h>
#endif

namespace sentinel::util {

namespace {

std::once_flag g_init_flag;
std::shared_ptr<spdlog::logger> g_logger;

spdlog::level::level_enum parse_level(std::string_view level) {
    if (level == "trace")    return spdlog::level::trace;
    if (level == "debug")    return spdlog::level::debug;
    if (level == "info")     return spdlog::level::info;
    if (level == "warn")     return spdlog::level::warn;
    if (level == "warning")  return spdlog::level::warn;
    if (level == "error")    return spdlog::level::err;
    if (level == "critical") return spdlog::level::critical;
    if (level == "off")      return spdlog::level::off;
    return spdlog::level::info;
}

}  // namespace

void initialize_logging(const LogConfig& config) {
    std::call_once(g_init_flag, [&config]() {
        std::vector<spdlog::sink_ptr> sinks;

        // Console sink.
        if (config.console_enabled) {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] [%s:%#] %v");
            sinks.push_back(std::move(console_sink));
        }

        // Rotating file sink.
        if (!config.file_path.empty()) {
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                config.file_path,
                config.max_file_size,
                config.max_files
            );
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%s:%#] %v");
            sinks.push_back(std::move(file_sink));
        }

#ifdef __linux__
        // Syslog sink (Linux only).
        if (config.syslog_enabled) {
            auto syslog_sink = std::make_shared<spdlog::sinks::syslog_sink_mt>(
                config.syslog_ident, LOG_PID, LOG_USER, false
            );
            sinks.push_back(std::move(syslog_sink));
        }
#endif

        // Create the logger with all configured sinks.
        g_logger = std::make_shared<spdlog::logger>(
            "sentinel", sinks.begin(), sinks.end()
        );
        g_logger->set_level(parse_level(config.level));
        g_logger->flush_on(spdlog::level::warn);

        // Register as the default logger so that SPDLOG_* macros work.
        spdlog::set_default_logger(g_logger);
    });
}

void shutdown_logging() {
    if (g_logger) {
        g_logger->flush();
    }
    spdlog::shutdown();
}

std::shared_ptr<spdlog::logger> get_logger() {
    if (g_logger) {
        return g_logger;
    }
    // Fall back to spdlog default before initialization.
    return spdlog::default_logger();
}

void set_log_level(std::string_view level) {
    if (g_logger) {
        g_logger->set_level(parse_level(level));
    }
}

}  // namespace sentinel::util
