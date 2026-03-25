/// @file grpc_client.cpp
/// @brief gRPC client implementation for Correlation Engine communication.

#include "agent/grpc_client.h"
#include "util/crypto.h"
#include "util/logging.h"
#include "util/metrics.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <google/protobuf/timestamp.pb.h>

#include "sentinel/v1/probe.grpc.pb.h"
#include "sentinel/v1/telemetry.grpc.pb.h"

namespace sentinel::agent {

namespace {

/// Convert a system_clock time_point to a protobuf Timestamp.
void to_proto_timestamp(std::chrono::system_clock::time_point tp,
                         google::protobuf::Timestamp* out) {
    auto epoch = tp.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch);
    auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch - seconds);
    out->set_seconds(seconds.count());
    out->set_nanos(static_cast<int32_t>(nanos.count()));
}

/// Convert ProbeType enum to protobuf.
::sentinel::v1::ProbeType to_proto_probe_type(probes::ProbeType t) {
    switch (t) {
        case probes::ProbeType::kFma:           return ::sentinel::v1::PROBE_TYPE_FMA;
        case probes::ProbeType::kTensorCore:    return ::sentinel::v1::PROBE_TYPE_TENSOR_CORE;
        case probes::ProbeType::kTranscendental:return ::sentinel::v1::PROBE_TYPE_TRANSCENDENTAL;
        case probes::ProbeType::kAes:           return ::sentinel::v1::PROBE_TYPE_AES;
        case probes::ProbeType::kMemory:        return ::sentinel::v1::PROBE_TYPE_MEMORY;
        case probes::ProbeType::kRegisterFile:  return ::sentinel::v1::PROBE_TYPE_REGISTER_FILE;
        case probes::ProbeType::kSharedMemory:  return ::sentinel::v1::PROBE_TYPE_SHARED_MEMORY;
    }
    return ::sentinel::v1::PROBE_TYPE_UNSPECIFIED;
}

/// Convert protobuf ProbeType to our enum.
probes::ProbeType from_proto_probe_type(::sentinel::v1::ProbeType t) {
    switch (t) {
        case ::sentinel::v1::PROBE_TYPE_FMA:           return probes::ProbeType::kFma;
        case ::sentinel::v1::PROBE_TYPE_TENSOR_CORE:    return probes::ProbeType::kTensorCore;
        case ::sentinel::v1::PROBE_TYPE_TRANSCENDENTAL: return probes::ProbeType::kTranscendental;
        case ::sentinel::v1::PROBE_TYPE_AES:            return probes::ProbeType::kAes;
        case ::sentinel::v1::PROBE_TYPE_MEMORY:         return probes::ProbeType::kMemory;
        case ::sentinel::v1::PROBE_TYPE_REGISTER_FILE:  return probes::ProbeType::kRegisterFile;
        case ::sentinel::v1::PROBE_TYPE_SHARED_MEMORY:  return probes::ProbeType::kSharedMemory;
        default: return probes::ProbeType::kFma;
    }
}

}  // namespace

// ── Impl ─────────────────────────────────────────────────────────────

struct GrpcClient::Impl {
    GrpcConfig config;
    std::string hostname;
    std::string hmac_key;

    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<::sentinel::v1::ProbeService::Stub> probe_stub;
    std::unique_ptr<::sentinel::v1::TelemetryService::Stub> telemetry_stub;

    std::atomic<bool> running{false};
    std::atomic<bool> connected{false};
    std::atomic<uint64_t> sequence_num{0};

    // Probe result buffer.
    mutable std::mutex probe_mutex;
    std::condition_variable probe_cv;
    std::deque<probes::ProbeResult> probe_buffer;

    // Telemetry buffer.
    mutable std::mutex telemetry_mutex;
    std::condition_variable telemetry_cv;
    std::deque<std::vector<telemetry::NvmlSnapshot>> telemetry_buffer;

    // Callbacks.
    ScheduleOverrideCallback override_callback;

    // Threads.
    std::thread probe_sender_thread;
    std::thread telemetry_sender_thread;

    // Backoff state.
    std::chrono::seconds current_backoff{1};

    /// Create the gRPC channel with appropriate credentials.
    void create_channel() {
        grpc::ChannelArguments args;
        args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS,
                    static_cast<int>(config.keepalive_interval.count() * 1000));
        args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
        args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
        args.SetInt(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA, 0);

        std::shared_ptr<grpc::ChannelCredentials> creds;
        if (config.use_tls && !config.tls_ca_path.empty()) {
            grpc::SslCredentialsOptions ssl_opts;
            // In production, read cert/key/ca files here.
            creds = grpc::SslCredentials(ssl_opts);
        } else {
            creds = grpc::InsecureChannelCredentials();
        }

        channel = grpc::CreateCustomChannel(config.endpoint, creds, args);
        probe_stub = ::sentinel::v1::ProbeService::NewStub(channel);
        telemetry_stub = ::sentinel::v1::TelemetryService::NewStub(channel);
    }

    /// Build a ProbeExecution protobuf from a ProbeResult.
    ::sentinel::v1::ProbeExecution build_execution(const probes::ProbeResult& result) {
        ::sentinel::v1::ProbeExecution exec;
        exec.set_execution_id(std::to_string(result.probe_id));
        exec.set_probe_type(to_proto_probe_type(result.probe_type));

        auto* sm = exec.mutable_sm();
        auto* gpu = sm->mutable_gpu();
        gpu->set_uuid(result.gpu_uuid);
        gpu->set_hostname(hostname);
        sm->set_sm_id(result.sm_id);

        if (result.match) {
            exec.set_result(::sentinel::v1::PROBE_RESULT_PASS);
        } else {
            exec.set_result(::sentinel::v1::PROBE_RESULT_FAIL);
        }

        exec.set_expected_hash(result.expected_hash.data(), result.expected_hash.size());
        exec.set_actual_hash(result.actual_hash.data(), result.actual_hash.size());

        // Populate mismatch detail (first one only in proto).
        if (!result.mismatch_details.empty()) {
            auto* detail = exec.mutable_mismatch_detail();
            const auto& md = result.mismatch_details[0];
            detail->set_byte_offset(md.byte_offset);
            detail->set_expected_value(
                std::string(md.expected_value.begin(), md.expected_value.end()));
            detail->set_actual_value(
                std::string(md.actual_value.begin(), md.actual_value.end()));
            for (uint32_t bit : md.differing_bits) {
                detail->add_differing_bits(bit);
            }
        }

        exec.set_execution_time_ns(result.execution_time_ns);
        exec.set_gpu_clock_mhz(result.gpu_clock_mhz);
        exec.set_gpu_temperature_c(result.gpu_temp_c);
        exec.set_gpu_power_w(result.gpu_power_w);

        to_proto_timestamp(result.timestamp, exec.mutable_timestamp());

        // HMAC signature.
        exec.set_hmac_signature(result.hmac.data(), result.hmac.size());

        return exec;
    }

    /// Probe sender thread: batches results and streams them.
    void probe_sender_loop() {
        SENTINEL_LOG_INFO("Probe sender thread started");

        while (running.load()) {
            // Wait for data or timeout.
            std::unique_lock lock(probe_mutex);
            probe_cv.wait_for(lock, config.batch_timeout,
                               [this] {
                                   return !probe_buffer.empty() || !running.load();
                               });

            if (!running.load() && probe_buffer.empty()) break;

            // Collect a batch.
            std::vector<probes::ProbeResult> batch;
            batch.reserve(config.max_batch_size);
            while (!probe_buffer.empty() && batch.size() < config.max_batch_size) {
                batch.push_back(std::move(probe_buffer.front()));
                probe_buffer.pop_front();
            }
            lock.unlock();

            if (batch.empty()) continue;

            // Attempt to send.
            if (!send_probe_batch(batch)) {
                // On failure, re-enqueue for retry.
                lock.lock();
                for (auto& r : batch) {
                    probe_buffer.push_back(std::move(r));
                }

                // Enforce buffer capacity (1 hour at ~100 results/sec).
                constexpr std::size_t max_buffer = 360000;
                while (probe_buffer.size() > max_buffer) {
                    probe_buffer.pop_front();  // Drop oldest.
                }
                lock.unlock();

                // Exponential backoff.
                SENTINEL_LOG_WARN("gRPC send failed, backing off for {}s",
                                  current_backoff.count());
                std::this_thread::sleep_for(current_backoff);
                current_backoff = std::min(current_backoff * 2, config.max_backoff);
            } else {
                current_backoff = config.initial_backoff;
            }
        }

        SENTINEL_LOG_INFO("Probe sender thread stopped");
    }

    /// Send a batch of probe results via the bidirectional stream.
    bool send_probe_batch(const std::vector<probes::ProbeResult>& batch) {
        try {
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() +
                                  std::chrono::seconds(30));

            auto stream = probe_stub->StreamProbeResults(&context);
            if (!stream) {
                connected.store(false);
                util::Metrics::instance().set_grpc_connected(false);
                util::Metrics::instance().record_grpc_error("stream_create_failed");
                return false;
            }

            connected.store(true);
            util::Metrics::instance().set_grpc_connected(true);

            // Build batch message.
            ::sentinel::v1::ProbeResultBatch proto_batch;
            proto_batch.set_agent_hostname(hostname);
            proto_batch.set_sequence_number(
                sequence_num.fetch_add(1, std::memory_order_relaxed));
            to_proto_timestamp(std::chrono::system_clock::now(),
                               proto_batch.mutable_batch_timestamp());

            for (const auto& result : batch) {
                *proto_batch.add_executions() = build_execution(result);
            }

            // Send the batch.
            if (!stream->Write(proto_batch)) {
                SENTINEL_LOG_ERROR("Failed to write probe batch to stream");
                util::Metrics::instance().record_grpc_error("write_failed");
                stream->WritesDone();
                stream->Finish();
                connected.store(false);
                return false;
            }

            stream->WritesDone();

            // Read ack.
            ::sentinel::v1::ProbeAck ack;
            if (stream->Read(&ack)) {
                if (!ack.accepted()) {
                    SENTINEL_LOG_WARN("Probe batch rejected: {}", ack.rejection_reason());
                }

                // Process schedule overrides.
                if (override_callback) {
                    for (const auto& override_msg : ack.schedule_overrides()) {
                        override_callback(
                            from_proto_probe_type(override_msg.probe_type()),
                            override_msg.period_seconds(),
                            override_msg.duration_seconds());
                    }
                }
            }

            grpc::Status status = stream->Finish();
            if (!status.ok()) {
                SENTINEL_LOG_ERROR("gRPC stream finish error: {} ({})",
                                    status.error_message(),
                                    static_cast<int>(status.error_code()));
                util::Metrics::instance().record_grpc_error("finish_error");
                connected.store(false);
                return false;
            }

            util::Metrics::instance().record_batch_sent();
            return true;

        } catch (const std::exception& e) {
            SENTINEL_LOG_ERROR("gRPC send exception: {}", e.what());
            util::Metrics::instance().record_grpc_error("exception");
            connected.store(false);
            return false;
        }
    }

    /// Telemetry sender thread.
    void telemetry_sender_loop() {
        SENTINEL_LOG_INFO("Telemetry sender thread started");

        while (running.load()) {
            std::unique_lock lock(telemetry_mutex);
            telemetry_cv.wait_for(lock, std::chrono::seconds(5),
                                   [this] {
                                       return !telemetry_buffer.empty() || !running.load();
                                   });

            if (!running.load() && telemetry_buffer.empty()) break;

            std::vector<std::vector<telemetry::NvmlSnapshot>> batches;
            while (!telemetry_buffer.empty()) {
                batches.push_back(std::move(telemetry_buffer.front()));
                telemetry_buffer.pop_front();
            }
            lock.unlock();

            for (const auto& snapshots : batches) {
                send_telemetry_batch(snapshots);
            }
        }

        SENTINEL_LOG_INFO("Telemetry sender thread stopped");
    }

    /// Send a telemetry batch.
    void send_telemetry_batch(const std::vector<telemetry::NvmlSnapshot>& snapshots) {
        try {
            grpc::ClientContext context;
            context.set_deadline(std::chrono::system_clock::now() +
                                  std::chrono::seconds(10));

            auto stream = telemetry_stub->StreamTelemetry(&context);
            if (!stream) return;

            ::sentinel::v1::TelemetryBatch batch;
            batch.set_agent_hostname(hostname);
            batch.set_sequence_number(
                sequence_num.fetch_add(1, std::memory_order_relaxed));
            to_proto_timestamp(std::chrono::system_clock::now(),
                               batch.mutable_batch_timestamp());

            for (const auto& snap : snapshots) {
                auto* report = batch.add_reports();

                auto* thermal = report->mutable_thermal();
                auto* gpu_id = thermal->mutable_gpu();
                gpu_id->set_uuid(snap.gpu_uuid);
                gpu_id->set_hostname(hostname);
                gpu_id->set_device_index(snap.device_index);
                thermal->set_temperature_c(snap.temperature_c);
                thermal->set_fan_speed_pct(snap.fan_speed_pct);
                thermal->set_throttle_active(snap.throttle_active);
                thermal->set_memory_temperature_c(snap.memory_temperature_c);

                auto* power = report->mutable_power();
                power->mutable_gpu()->CopyFrom(*gpu_id);
                power->set_power_w(snap.power_w);
                power->set_voltage_mv(snap.voltage_mv);
                power->set_power_limit_w(snap.power_limit_w);
                power->set_power_throttle_active(snap.power_throttle_active);

                auto* ecc = report->mutable_ecc();
                ecc->mutable_gpu()->CopyFrom(*gpu_id);
                ecc->set_sram_corrected(snap.sram_corrected);
                ecc->set_sram_uncorrected(snap.sram_uncorrected);
                ecc->set_dram_corrected(snap.dram_corrected);
                ecc->set_dram_uncorrected(snap.dram_uncorrected);
                ecc->set_retired_pages(snap.retired_pages);
                ecc->set_pending_retired_pages(snap.pending_retired_pages);
                ecc->set_reset_required(snap.reset_required);

                report->set_gpu_utilization_pct(snap.gpu_utilization_pct);
                report->set_memory_utilization_pct(snap.memory_utilization_pct);
                report->set_gpu_clock_mhz(snap.gpu_clock_mhz);
                report->set_mem_clock_mhz(snap.mem_clock_mhz);
                report->set_pcie_gen(snap.pcie_gen);
                report->set_pcie_width(snap.pcie_width);
            }

            stream->Write(batch);
            stream->WritesDone();

            ::sentinel::v1::TelemetryAck ack;
            stream->Read(&ack);
            stream->Finish();

        } catch (const std::exception& e) {
            SENTINEL_LOG_WARN("Telemetry send failed: {}", e.what());
        }
    }
};

// ── Public interface ────────────────────────────────────────────────

GrpcClient::GrpcClient() : impl_(std::make_unique<Impl>()) {}
GrpcClient::~GrpcClient() { stop(); }

bool GrpcClient::initialize(const GrpcConfig& config,
                              const std::string& hostname,
                              const std::string& hmac_key) {
    impl_->config = config;
    impl_->hostname = hostname;
    impl_->hmac_key = hmac_key;

    try {
        impl_->create_channel();
        SENTINEL_LOG_INFO("gRPC client initialized, endpoint: {}", config.endpoint);
        return true;
    } catch (const std::exception& e) {
        SENTINEL_LOG_ERROR("gRPC client initialization failed: {}", e.what());
        return false;
    }
}

void GrpcClient::start() {
    if (impl_->running.load()) return;
    impl_->running.store(true);

    impl_->probe_sender_thread = std::thread(&Impl::probe_sender_loop, impl_.get());
    impl_->telemetry_sender_thread = std::thread(&Impl::telemetry_sender_loop, impl_.get());

    SENTINEL_LOG_INFO("gRPC client started");
}

void GrpcClient::stop() {
    if (!impl_->running.load()) return;

    impl_->running.store(false);
    impl_->probe_cv.notify_all();
    impl_->telemetry_cv.notify_all();

    if (impl_->probe_sender_thread.joinable()) {
        impl_->probe_sender_thread.join();
    }
    if (impl_->telemetry_sender_thread.joinable()) {
        impl_->telemetry_sender_thread.join();
    }

    SENTINEL_LOG_INFO("gRPC client stopped");
}

void GrpcClient::send_probe_result(probes::ProbeResult result) {
    // Sign the result with HMAC.
    util::Sha256Hasher hasher;
    hasher.update(&result.probe_id, sizeof(result.probe_id));
    hasher.update(&result.probe_type, sizeof(result.probe_type));
    hasher.update(reinterpret_cast<const uint8_t*>(result.gpu_uuid.data()),
                  result.gpu_uuid.size());
    hasher.update(&result.sm_id, sizeof(result.sm_id));
    hasher.update(result.expected_hash.data(), result.expected_hash.size());
    hasher.update(result.actual_hash.data(), result.actual_hash.size());
    auto payload_hash = hasher.finalize();

    result.hmac = util::hmac_sha256(impl_->hmac_key,
                                     std::span<const uint8_t>(payload_hash));

    {
        std::lock_guard lock(impl_->probe_mutex);
        impl_->probe_buffer.push_back(std::move(result));
    }
    impl_->probe_cv.notify_one();

    util::Metrics::instance().set_buffer_depth(impl_->probe_buffer.size());
}

void GrpcClient::send_telemetry(const std::vector<telemetry::NvmlSnapshot>& snapshots) {
    {
        std::lock_guard lock(impl_->telemetry_mutex);
        impl_->telemetry_buffer.push_back(snapshots);
    }
    impl_->telemetry_cv.notify_one();
}

void GrpcClient::on_schedule_override(ScheduleOverrideCallback callback) {
    impl_->override_callback = std::move(callback);
}

bool GrpcClient::is_connected() const {
    return impl_->connected.load();
}

std::size_t GrpcClient::buffered_count() const {
    std::lock_guard lock(impl_->probe_mutex);
    return impl_->probe_buffer.size();
}

uint64_t GrpcClient::sequence_number() const {
    return impl_->sequence_num.load();
}

}  // namespace sentinel::agent
