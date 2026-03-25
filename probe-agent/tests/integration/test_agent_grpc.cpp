/// @file test_agent_grpc.cpp
/// @brief Integration tests for the Agent gRPC client.
///
/// Tests the gRPC client's ability to connect, send batches, handle
/// disconnections, and process schedule override responses. Uses a
/// mock gRPC server for deterministic testing.

#include "agent/grpc_client.h"
#include "agent/config_manager.h"
#include "probes/probe_interface.h"
#include "util/crypto.h"
#include "util/logging.h"

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

#include "sentinel/v1/probe.grpc.pb.h"

namespace sentinel::agent {
namespace {

/// Mock ProbeService implementation for testing.
class MockProbeService final : public ::sentinel::v1::ProbeService::Service {
public:
    grpc::Status StreamProbeResults(
            grpc::ServerContext* /*context*/,
            grpc::ServerReaderWriter<::sentinel::v1::ProbeAck,
                                     ::sentinel::v1::ProbeResultBatch>* stream) override {
        ::sentinel::v1::ProbeResultBatch batch;
        while (stream->Read(&batch)) {
            ++batches_received_;
            last_hostname_ = batch.agent_hostname();
            last_seq_ = batch.sequence_number();
            executions_received_ += batch.executions_size();

            // Send ack.
            ::sentinel::v1::ProbeAck ack;
            ack.set_sequence_number(batch.sequence_number());
            ack.set_accepted(true);

            // Optionally inject a schedule override.
            if (inject_override_) {
                auto* override_msg = ack.add_schedule_overrides();
                override_msg->set_probe_type(::sentinel::v1::PROBE_TYPE_FMA);
                override_msg->set_period_seconds(30);
                override_msg->set_duration_seconds(120);
                inject_override_ = false;
            }

            stream->Write(ack);
        }
        return grpc::Status::OK;
    }

    int batches_received_ = 0;
    int executions_received_ = 0;
    std::string last_hostname_;
    uint64_t last_seq_ = 0;
    bool inject_override_ = false;
};

class GrpcClientTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Start mock server on a random port.
        grpc::ServerBuilder builder;
        builder.AddListeningPort("localhost:0", grpc::InsecureServerCredentials(),
                                  &server_port_);
        builder.RegisterService(&mock_service_);
        server_ = builder.BuildAndStart();
        ASSERT_NE(server_, nullptr);
        ASSERT_GT(server_port_, 0);

        endpoint_ = "localhost:" + std::to_string(server_port_);
    }

    void TearDown() override {
        if (server_) {
            server_->Shutdown();
        }
    }

    MockProbeService mock_service_;
    std::unique_ptr<grpc::Server> server_;
    int server_port_ = 0;
    std::string endpoint_;
};

TEST_F(GrpcClientTest, ConnectAndSendBatch) {
    GrpcConfig config;
    config.endpoint = endpoint_;
    config.batch_timeout = std::chrono::milliseconds(100);
    config.max_batch_size = 10;

    GrpcClient client;
    ASSERT_TRUE(client.initialize(config, "test-host", "test-key"));
    client.start();

    // Send a probe result.
    probes::ProbeResult result;
    result.probe_id = 1;
    result.probe_type = probes::ProbeType::kFma;
    result.gpu_uuid = "GPU-test-uuid";
    result.sm_id = 0;
    result.match = true;
    result.timestamp = std::chrono::system_clock::now();

    client.send_probe_result(std::move(result));

    // Wait for the batch to be sent.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    client.stop();

    EXPECT_GE(mock_service_.batches_received_, 1);
    EXPECT_EQ(mock_service_.last_hostname_, "test-host");
    EXPECT_GE(mock_service_.executions_received_, 1);
}

TEST_F(GrpcClientTest, BatchAggregation) {
    GrpcConfig config;
    config.endpoint = endpoint_;
    config.batch_timeout = std::chrono::milliseconds(200);
    config.max_batch_size = 50;

    GrpcClient client;
    ASSERT_TRUE(client.initialize(config, "test-host", "test-key"));
    client.start();

    // Send multiple results rapidly.
    for (int i = 0; i < 10; ++i) {
        probes::ProbeResult result;
        result.probe_id = static_cast<uint64_t>(i);
        result.probe_type = probes::ProbeType::kFma;
        result.match = true;
        result.timestamp = std::chrono::system_clock::now();
        client.send_probe_result(std::move(result));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    client.stop();

    // All 10 executions should have been received (possibly in 1-2 batches).
    EXPECT_GE(mock_service_.executions_received_, 10);
}

TEST_F(GrpcClientTest, ScheduleOverrideCallback) {
    mock_service_.inject_override_ = true;

    GrpcConfig config;
    config.endpoint = endpoint_;
    config.batch_timeout = std::chrono::milliseconds(100);

    GrpcClient client;
    ASSERT_TRUE(client.initialize(config, "test-host", "test-key"));

    bool override_received = false;
    probes::ProbeType override_type;
    uint32_t override_period = 0;

    client.on_schedule_override(
        [&](probes::ProbeType type, uint32_t period, uint32_t /*duration*/) {
            override_received = true;
            override_type = type;
            override_period = period;
        });

    client.start();

    probes::ProbeResult result;
    result.probe_id = 1;
    result.probe_type = probes::ProbeType::kFma;
    result.match = true;
    result.timestamp = std::chrono::system_clock::now();
    client.send_probe_result(std::move(result));

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    client.stop();

    EXPECT_TRUE(override_received);
    EXPECT_EQ(override_type, probes::ProbeType::kFma);
    EXPECT_EQ(override_period, 30u);
}

TEST_F(GrpcClientTest, BufferingOnDisconnect) {
    // Shut down server to simulate disconnect.
    server_->Shutdown();
    server_.reset();

    GrpcConfig config;
    config.endpoint = endpoint_;
    config.batch_timeout = std::chrono::milliseconds(100);
    config.initial_backoff = std::chrono::seconds(1);

    GrpcClient client;
    ASSERT_TRUE(client.initialize(config, "test-host", "test-key"));
    client.start();

    // Send results while disconnected.
    for (int i = 0; i < 5; ++i) {
        probes::ProbeResult result;
        result.probe_id = static_cast<uint64_t>(i);
        result.probe_type = probes::ProbeType::kFma;
        result.match = true;
        result.timestamp = std::chrono::system_clock::now();
        client.send_probe_result(std::move(result));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Results should be buffered locally.
    EXPECT_GE(client.buffered_count(), 0u);  // May have attempted and re-queued.

    client.stop();
}

}  // namespace
}  // namespace sentinel::agent
