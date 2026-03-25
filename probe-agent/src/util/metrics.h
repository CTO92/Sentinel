/// @file metrics.h
/// @brief Prometheus metrics exposition for the probe agent.
///
/// Exposes counters, gauges, and histograms for probe execution, latency,
/// errors, telemetry collection, and gRPC client health. The HTTP endpoint
/// is served on a configurable port (default 9101).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace prometheus {
class Registry;
class Counter;
class Gauge;
class Histogram;
class Family;
template <typename T>
class Family;
}  // namespace prometheus

namespace sentinel::util {

/// Configuration for the Prometheus metrics exporter.
struct MetricsConfig {
    /// HTTP port to expose /metrics on.
    uint16_t port = 9101;

    /// Bind address (e.g., "0.0.0.0").
    std::string bind_address = "0.0.0.0";

    /// Whether metrics exposition is enabled.
    bool enabled = true;
};

/// Singleton metrics registry for the probe agent.
class Metrics {
public:
    /// Initialize the metrics system. Must be called once at startup.
    static void initialize(const MetricsConfig& config);

    /// Shut down the HTTP server and clean up.
    static void shutdown();

    /// Get the shared metrics instance. Asserts that initialize() was called.
    static Metrics& instance();

    // ── Probe metrics ─────────────────────────────────────────────────

    /// Increment the probe execution counter.
    /// @param probe_type  e.g., "fma", "tensor_core"
    /// @param result      "pass", "fail", "error", "timeout"
    void record_probe_execution(std::string_view probe_type, std::string_view result);

    /// Record probe execution latency in seconds.
    void record_probe_latency(std::string_view probe_type, double seconds);

    /// Set the number of SMs currently under test on a given GPU.
    void set_active_sm_count(std::string_view gpu_uuid, int count);

    // ── gRPC metrics ──────────────────────────────────────────────────

    /// Increment the number of probe batches sent.
    void record_batch_sent();

    /// Increment the number of gRPC errors.
    void record_grpc_error(std::string_view error_type);

    /// Set the gRPC connection state (1 = connected, 0 = disconnected).
    void set_grpc_connected(bool connected);

    /// Set the local buffer depth (results pending send).
    void set_buffer_depth(std::size_t depth);

    // ── Telemetry metrics ─────────────────────────────────────────────

    /// Record a telemetry collection cycle.
    void record_telemetry_collection();

    /// Record a telemetry collection error.
    void record_telemetry_error(std::string_view source);

    // ── System metrics ────────────────────────────────────────────────

    /// Set agent uptime in seconds.
    void set_uptime(double seconds);

    /// Set the number of GPUs managed by this agent.
    void set_gpu_count(int count);

private:
    Metrics() = default;

    struct Impl;
    std::unique_ptr<Impl> impl_;

    static Metrics* s_instance;
};

}  // namespace sentinel::util
