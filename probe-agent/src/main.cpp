/// @file main.cpp
/// @brief Entry point for the SENTINEL Probe Agent daemon.
///
/// Parses command-line arguments, sets up signal handlers for graceful
/// shutdown, and launches the Agent.

#include "agent/agent.h"
#include "util/logging.h"

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

/// Global agent pointer for signal handler access.
std::atomic<sentinel::agent::Agent*> g_agent{nullptr};

/// Signal handler for SIGINT / SIGTERM.
void signal_handler(int sig) {
    const char* sig_name = (sig == SIGINT) ? "SIGINT" : "SIGTERM";
    // Cannot use spdlog in signal handler; use write() for async-signal-safety.
    const char msg[] = "\n[sentinel] Received shutdown signal\n";
    [[maybe_unused]] auto r = write(STDERR_FILENO, msg, sizeof(msg) - 1);

    auto* agent = g_agent.load(std::memory_order_relaxed);
    if (agent) {
        agent->request_shutdown();
    }
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0 << " [options]\n"
              << "\n"
              << "SENTINEL Probe Agent — GPU Silent Data Corruption detector\n"
              << "\n"
              << "Options:\n"
              << "  -c, --config <path>   Path to JSON configuration file\n"
              << "  -h, --help            Show this help message\n"
              << "  -v, --version         Show version information\n"
              << "\n"
              << "Environment variables:\n"
              << "  SENTINEL_CONFIG       Path to config file (overridden by -c)\n"
              << "  SENTINEL_LOG_LEVEL    Log level (trace/debug/info/warn/error)\n"
              << "  SENTINEL_GRPC_ENDPOINT  gRPC endpoint (host:port)\n"
              << std::endl;
}

void print_version() {
    std::cout << "sentinel-probe-agent v0.1.0\n"
              << "Built with CUDA " << __CUDACC_VER_MAJOR__ << "."
              << __CUDACC_VER_MINOR__ << "\n"
              << "C++ standard: " << __cplusplus << "\n"
              << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    // Parse command-line arguments.
    std::string config_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (arg == "-v" || arg == "--version") {
            print_version();
            return EXIT_SUCCESS;
        }
        if ((arg == "-c" || arg == "--config") && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    // Check environment variable for config path.
    if (config_path.empty()) {
        const char* env_config = std::getenv("SENTINEL_CONFIG");
        if (env_config) {
            config_path = env_config;
        }
    }

    // Install signal handlers.
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Create and initialize the agent.
    sentinel::agent::Agent agent;
    g_agent.store(&agent, std::memory_order_relaxed);

    if (!agent.initialize(config_path)) {
        std::cerr << "[sentinel] Agent initialization failed. Exiting.\n";
        return EXIT_FAILURE;
    }

    // Run until shutdown signal.
    agent.run();

    g_agent.store(nullptr, std::memory_order_relaxed);
    return EXIT_SUCCESS;
}
