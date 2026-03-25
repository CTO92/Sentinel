/// @file metrics.cpp
/// @brief Prometheus metrics exposition implementation.

#include "util/metrics.h"
#include "util/logging.h"

#include <cassert>
#include <mutex>

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/gauge.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

namespace sentinel::util {

struct Metrics::Impl {
    std::shared_ptr<prometheus::Registry> registry;
    std::unique_ptr<prometheus::Exposer> exposer;

    // Probe metrics.
    prometheus::Family<prometheus::Counter>& probe_executions;
    prometheus::Family<prometheus::Histogram>& probe_latency;
    prometheus::Family<prometheus::Gauge>& active_sms;

    // gRPC metrics.
    prometheus::Counter& batches_sent;
    prometheus::Family<prometheus::Counter>& grpc_errors;
    prometheus::Gauge& grpc_connected;
    prometheus::Gauge& buffer_depth;

    // Telemetry metrics.
    prometheus::Counter& telemetry_collections;
    prometheus::Family<prometheus::Counter>& telemetry_errors;

    // System metrics.
    prometheus::Gauge& uptime;
    prometheus::Gauge& gpu_count;

    explicit Impl(const MetricsConfig& config)
        : registry{std::make_shared<prometheus::Registry>()},
          // Probe families.
          probe_executions{prometheus::BuildCounter()
                               .Name("sentinel_probe_executions_total")
                               .Help("Total number of probe executions")
                               .Register(*registry)},
          probe_latency{prometheus::BuildHistogram()
                            .Name("sentinel_probe_latency_seconds")
                            .Help("Probe execution latency in seconds")
                            .Register(*registry)},
          active_sms{prometheus::BuildGauge()
                         .Name("sentinel_active_sm_count")
                         .Help("Number of SMs currently under test")
                         .Register(*registry)},
          // gRPC family.
          batches_sent{prometheus::BuildCounter()
                           .Name("sentinel_grpc_batches_sent_total")
                           .Help("Total probe batches sent via gRPC")
                           .Register(*registry)
                           .Add({})},
          grpc_errors{prometheus::BuildCounter()
                          .Name("sentinel_grpc_errors_total")
                          .Help("Total gRPC errors by type")
                          .Register(*registry)},
          grpc_connected{prometheus::BuildGauge()
                             .Name("sentinel_grpc_connected")
                             .Help("1 if gRPC stream is connected, 0 otherwise")
                             .Register(*registry)
                             .Add({})},
          buffer_depth{prometheus::BuildGauge()
                           .Name("sentinel_buffer_depth")
                           .Help("Number of results buffered pending send")
                           .Register(*registry)
                           .Add({})},
          // Telemetry family.
          telemetry_collections{prometheus::BuildCounter()
                                    .Name("sentinel_telemetry_collections_total")
                                    .Help("Total telemetry collection cycles")
                                    .Register(*registry)
                                    .Add({})},
          telemetry_errors{prometheus::BuildCounter()
                               .Name("sentinel_telemetry_errors_total")
                               .Help("Total telemetry collection errors")
                               .Register(*registry)},
          // System family.
          uptime{prometheus::BuildGauge()
                     .Name("sentinel_agent_uptime_seconds")
                     .Help("Agent uptime in seconds")
                     .Register(*registry)
                     .Add({})},
          gpu_count{prometheus::BuildGauge()
                        .Name("sentinel_gpu_count")
                        .Help("Number of GPUs managed by this agent")
                        .Register(*registry)
                        .Add({})} {
        if (config.enabled) {
            std::string endpoint = config.bind_address + ":" + std::to_string(config.port);
            exposer = std::make_unique<prometheus::Exposer>(endpoint);
            exposer->RegisterCollectable(registry);
            SENTINEL_LOG_INFO("Prometheus metrics exposed on {}", endpoint);
        }
    }
};

Metrics* Metrics::s_instance = nullptr;

void Metrics::initialize(const MetricsConfig& config) {
    static std::once_flag flag;
    std::call_once(flag, [&config]() {
        s_instance = new Metrics();
        s_instance->impl_ = std::make_unique<Impl>(config);
    });
}

void Metrics::shutdown() {
    if (s_instance) {
        s_instance->impl_.reset();
        delete s_instance;
        s_instance = nullptr;
    }
}

Metrics& Metrics::instance() {
    assert(s_instance && "Metrics::initialize() must be called first");
    return *s_instance;
}

void Metrics::record_probe_execution(std::string_view probe_type, std::string_view result) {
    impl_->probe_executions
        .Add({{"probe_type", std::string(probe_type)}, {"result", std::string(result)}})
        .Increment();
}

void Metrics::record_probe_latency(std::string_view probe_type, double seconds) {
    impl_->probe_latency
        .Add({{"probe_type", std::string(probe_type)}},
             prometheus::Histogram::BucketBoundaries{
                 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0})
        .Observe(seconds);
}

void Metrics::set_active_sm_count(std::string_view gpu_uuid, int count) {
    impl_->active_sms
        .Add({{"gpu_uuid", std::string(gpu_uuid)}})
        .Set(static_cast<double>(count));
}

void Metrics::record_batch_sent() {
    impl_->batches_sent.Increment();
}

void Metrics::record_grpc_error(std::string_view error_type) {
    impl_->grpc_errors
        .Add({{"error_type", std::string(error_type)}})
        .Increment();
}

void Metrics::set_grpc_connected(bool connected) {
    impl_->grpc_connected.Set(connected ? 1.0 : 0.0);
}

void Metrics::set_buffer_depth(std::size_t depth) {
    impl_->buffer_depth.Set(static_cast<double>(depth));
}

void Metrics::record_telemetry_collection() {
    impl_->telemetry_collections.Increment();
}

void Metrics::record_telemetry_error(std::string_view source) {
    impl_->telemetry_errors
        .Add({{"source", std::string(source)}})
        .Increment();
}

void Metrics::set_uptime(double seconds) {
    impl_->uptime.Set(seconds);
}

void Metrics::set_gpu_count(int count) {
    impl_->gpu_count.Set(static_cast<double>(count));
}

}  // namespace sentinel::util
