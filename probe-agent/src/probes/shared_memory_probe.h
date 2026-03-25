/// @file shared_memory_probe.h
/// @brief Per-SM shared memory integrity probe.
///
/// Tests the shared memory (SMEM) attached to each SM by writing
/// deterministic patterns, reading them back, and verifying correctness.
/// This catches stuck-at faults and addressing errors in the shared
/// memory banks.

#pragma once

#include "probes/probe_interface.h"

#include <memory>

namespace sentinel::probes {

class SharedMemoryProbe : public ProbeInterface {
public:
    SharedMemoryProbe();
    ~SharedMemoryProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kSharedMemory; }
    [[nodiscard]] std::string_view name() const noexcept override { return "shared_memory"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 120; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kMedium; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kAll; }

    /// Shared memory test size in bytes (48 KB — typical max per block).
    static constexpr std::size_t kSmemTestBytes = 48 * 1024;

    /// Number of 32-bit words in the test region.
    static constexpr uint32_t kSmemTestWords = kSmemTestBytes / sizeof(uint32_t);

    /// Number of test patterns.
    static constexpr uint32_t kNumPatterns = 6;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
