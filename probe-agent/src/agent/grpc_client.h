/// @file grpc_client.h
/// @brief Asynchronous gRPC client for bidirectional streaming to the
///        Correlation Engine.
///
/// Manages the ProbeService::StreamProbeResults and
/// TelemetryService::StreamTelemetry bidirectional streams. Handles
/// connection lifecycle, exponential backoff on failure, local buffering
/// up to 1 hour of results, and HMAC signing of outgoing batches.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "agent/config_manager.h"
#include "probes/probe_interface.h"
#include "telemetry/nvml_collector.h"

namespace sentinel::agent {

/// Callback for schedule overrides received from the server.
using ScheduleOverrideCallback =
    std::function<void(probes::ProbeType type, uint32_t period_s, uint32_t duration_s)>;

/// Asynchronous gRPC client for the Correlation Engine.
class GrpcClient {
public:
    GrpcClient();
    ~GrpcClient();

    GrpcClient(const GrpcClient&) = delete;
    GrpcClient& operator=(const GrpcClient&) = delete;

    /// Initialize the client with configuration.
    /// @param config gRPC configuration.
    /// @param hostname Agent hostname for batch identification.
    /// @param hmac_key Key for HMAC signing.
    /// @return true if initialization succeeded.
    bool initialize(const GrpcConfig& config,
                     const std::string& hostname,
                     const std::string& hmac_key);

    /// Start the gRPC client threads (sender + receiver).
    void start();

    /// Stop the client and drain any pending sends.
    void stop();

    /// Enqueue a probe result for batching and transmission.
    /// Thread-safe; can be called from any executor thread.
    void send_probe_result(probes::ProbeResult result);

    /// Enqueue a telemetry snapshot for transmission.
    void send_telemetry(const std::vector<telemetry::NvmlSnapshot>& snapshots);

    /// Register a callback for schedule override messages from the server.
    void on_schedule_override(ScheduleOverrideCallback callback);

    /// @return true if the gRPC stream is currently connected.
    [[nodiscard]] bool is_connected() const;

    /// @return Number of results currently buffered locally.
    [[nodiscard]] std::size_t buffered_count() const;

    /// @return Current sequence number.
    [[nodiscard]] uint64_t sequence_number() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::agent
