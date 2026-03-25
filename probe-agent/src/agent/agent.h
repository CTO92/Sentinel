/// @file agent.h
/// @brief Top-level Probe Agent orchestrator.
///
/// Owns and coordinates all subsystems: probe initialization, scheduling,
/// execution, telemetry collection, gRPC communication, and watchdog
/// monitoring. This is the main entry point for the probe agent daemon.

#pragma once

#include <memory>
#include <string>

namespace sentinel::agent {

/// The Probe Agent daemon.
class Agent {
public:
    Agent();
    ~Agent();

    Agent(const Agent&) = delete;
    Agent& operator=(const Agent&) = delete;

    /// Initialize all subsystems.
    /// @param config_path Path to the JSON configuration file. If empty,
    ///                    defaults are used.
    /// @return true if initialization succeeded for at least one GPU.
    bool initialize(const std::string& config_path = "");

    /// Start all threads and begin probe execution.
    void run();

    /// Signal the agent to stop gracefully.
    void request_shutdown();

    /// Block until the agent has stopped.
    void wait_for_shutdown();

    /// @return true if the agent is currently running.
    [[nodiscard]] bool is_running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::agent
