/// @file memory_probe.h
/// @brief GPU global memory integrity probe using March C- test algorithm.
///
/// Performs a March C- test on 16MB of global memory plus shared memory
/// per SM. March C- is a well-known memory test that detects:
///   - Stuck-at faults (SAF)
///   - Transition faults (TF)
///   - Coupling faults (CF)
///   - Address decoder faults (AF)

#pragma once

#include "probes/probe_interface.h"

#include <memory>

namespace sentinel::probes {

class MemoryProbe : public ProbeInterface {
public:
    MemoryProbe();
    ~MemoryProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kMemory; }
    [[nodiscard]] std::string_view name() const noexcept override { return "memory"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 600; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kLow; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kSample10Pct; }

    /// Size of the global memory test region in bytes (16 MB).
    static constexpr std::size_t kTestRegionSize = 16 * 1024 * 1024;

    /// Size in 32-bit words.
    static constexpr std::size_t kTestRegionWords = kTestRegionSize / sizeof(uint32_t);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
