/// @file grpc_client_noop.cpp
/// @brief No-op gRPC client implementation when protobuf/gRPC are unavailable.
///
/// Compiled only when SENTINEL_HAS_GRPC is NOT defined. All methods are
/// intentional no-ops so the rest of the codebase can use GrpcClient
/// unconditionally.

#include "agent/grpc_client.h"
#include "util/logging.h"

namespace sentinel::agent {

struct GrpcClient::Impl {};

GrpcClient::GrpcClient() : impl_(std::make_unique<Impl>()) {}
GrpcClient::~GrpcClient() = default;

bool GrpcClient::initialize(const GrpcConfig& /*config*/,
                             const std::string& /*hostname*/,
                             const std::string& /*hmac_key*/) {
    SENTINEL_LOG_INFO("gRPC client disabled (built without gRPC support)");
    return true;
}

void GrpcClient::start() {}
void GrpcClient::stop() {}

void GrpcClient::send_probe_result(probes::ProbeResult /*result*/) {}
void GrpcClient::send_telemetry(
    const std::vector<telemetry::NvmlSnapshot>& /*snapshots*/) {}

void GrpcClient::on_schedule_override(ScheduleOverrideCallback /*callback*/) {}

bool GrpcClient::is_connected() const { return false; }
std::size_t GrpcClient::buffered_count() const { return 0; }
uint64_t GrpcClient::sequence_number() const { return 0; }

}  // namespace sentinel::agent
