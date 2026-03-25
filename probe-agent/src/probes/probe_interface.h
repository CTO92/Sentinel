/// @file probe_interface.h
/// @brief Abstract interface for all GPU integrity probes.
///
/// Every probe type (FMA, Tensor Core, Memory, etc.) implements this
/// interface. The scheduler and executor invoke probes through this
/// polymorphic base class, enabling uniform lifecycle management.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>

#include "util/crypto.h"

namespace sentinel::probes {

/// Enumeration of probe types, mirroring the protobuf ProbeType enum.
enum class ProbeType : uint8_t {
    kFma           = 1,
    kTensorCore    = 2,
    kTranscendental = 3,
    kAes           = 4,
    kMemory        = 5,
    kRegisterFile  = 6,
    kSharedMemory  = 7,
};

/// Convert probe type to human-readable string.
[[nodiscard]] constexpr const char* probe_type_name(ProbeType t) {
    switch (t) {
        case ProbeType::kFma:           return "fma";
        case ProbeType::kTensorCore:    return "tensor_core";
        case ProbeType::kTranscendental: return "transcendental";
        case ProbeType::kAes:           return "aes";
        case ProbeType::kMemory:        return "memory";
        case ProbeType::kRegisterFile:  return "register_file";
        case ProbeType::kSharedMemory:  return "shared_memory";
    }
    return "unknown";
}

/// Scheduling priority levels.
enum class Priority : uint8_t {
    kLow    = 0,
    kMedium = 1,
    kHigh   = 2,
};

/// SM selection strategy for scheduling.
enum class SmSelection : uint8_t {
    kAll           = 0,   ///< Run on every SM.
    kSample25Pct   = 1,   ///< Random 25% of SMs each cycle.
    kSample10Pct   = 2,   ///< Random 10% of SMs each cycle.
};

/// Detailed information about a bit-level mismatch.
struct MismatchDetail {
    uint64_t byte_offset = 0;
    std::vector<uint8_t> expected_value;
    std::vector<uint8_t> actual_value;
    std::vector<uint32_t> differing_bits;
};

/// Result of a single probe execution on one SM.
struct ProbeResult {
    /// Unique probe execution ID (monotonically increasing).
    uint64_t probe_id = 0;

    /// Type of probe that was executed.
    ProbeType probe_type = ProbeType::kFma;

    /// UUID of the GPU where the probe ran.
    std::string gpu_uuid;

    /// SM index on which the probe executed.
    uint32_t sm_id = 0;

    /// SHA-256 of the expected (golden) output.
    util::Sha256Digest expected_hash{};

    /// SHA-256 of the actual output produced by the GPU.
    util::Sha256Digest actual_hash{};

    /// Whether expected and actual hashes match.
    bool match = false;

    /// Detailed mismatch information (populated only on failure).
    std::vector<MismatchDetail> mismatch_details;

    /// Kernel execution time in nanoseconds.
    uint64_t execution_time_ns = 0;

    /// GPU core clock at execution time (MHz).
    uint32_t gpu_clock_mhz = 0;

    /// GPU temperature at execution time (Celsius).
    float gpu_temp_c = 0.0f;

    /// GPU power draw at execution time (Watts).
    float gpu_power_w = 0.0f;

    /// Wall-clock timestamp when the probe completed.
    std::chrono::system_clock::time_point timestamp;

    /// HMAC-SHA256 signature over the result fields.
    util::Sha256Digest hmac{};
};

/// Abstract base class for all GPU integrity probes.
class ProbeInterface {
public:
    virtual ~ProbeInterface() = default;

    /// @return The type identifier for this probe.
    [[nodiscard]] virtual ProbeType type() const noexcept = 0;

    /// @return Human-readable name of this probe.
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Initialize the probe: allocate device/host buffers, upload golden
    /// data, etc. Called once per GPU at agent startup.
    ///
    /// @param device_index  CUDA device index.
    /// @return true on success.
    virtual bool initialize(int device_index) = 0;

    /// Execute the probe on a specific SM.
    ///
    /// @param sm_id   Target streaming multiprocessor index.
    /// @param stream  CUDA stream to launch the kernel on.
    /// @return        ProbeResult with pass/fail and timing data.
    [[nodiscard]] virtual ProbeResult execute(uint32_t sm_id,
                                               cudaStream_t stream) = 0;

    /// Release all device and host resources. Called at agent shutdown.
    virtual void teardown() = 0;

    /// @return Estimated device memory usage in bytes.
    [[nodiscard]] virtual std::size_t memory_footprint() const noexcept = 0;

    /// @return Default scheduling period in seconds.
    [[nodiscard]] virtual uint32_t default_period_seconds() const noexcept = 0;

    /// @return Default scheduling priority.
    [[nodiscard]] virtual Priority default_priority() const noexcept = 0;

    /// @return Default SM selection strategy.
    [[nodiscard]] virtual SmSelection default_sm_selection() const noexcept = 0;

protected:
    /// Helper: generate a unique probe ID (thread-safe, monotonically increasing).
    [[nodiscard]] static uint64_t next_probe_id();
};

/// Factory function to create all known probe types.
[[nodiscard]] std::vector<std::unique_ptr<ProbeInterface>> create_all_probes();

}  // namespace sentinel::probes
