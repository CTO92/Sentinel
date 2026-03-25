/// @file register_file_probe.h
/// @brief GPU register file integrity probe.
///
/// Writes known bit patterns (walking ones, walking zeros, alternating,
/// all-ones, all-zeros) to GPU registers via inline PTX and reads them
/// back. Detects stuck-at and coupling faults in the register file.

#pragma once

#include "probes/probe_interface.h"

#include <memory>

namespace sentinel::probes {

class RegisterFileProbe : public ProbeInterface {
public:
    RegisterFileProbe();
    ~RegisterFileProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kRegisterFile; }
    [[nodiscard]] std::string_view name() const noexcept override { return "register_file"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 120; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kMedium; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kAll; }

    /// Number of 32-bit register patterns to test.
    static constexpr uint32_t kNumPatterns = 96;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
