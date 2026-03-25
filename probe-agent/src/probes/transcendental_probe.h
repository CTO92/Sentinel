/// @file transcendental_probe.h
/// @brief Transcendental function accuracy probe.
///
/// Tests GPU Special Function Units (SFUs) by computing sin, cos, exp,
/// log, and rsqrt on 256 carefully chosen input values per function.
/// Results are compared against double-precision host reference with
/// 1 ULP tolerance.

#pragma once

#include "probes/probe_interface.h"

#include <memory>

namespace sentinel::probes {

class TranscendentalProbe : public ProbeInterface {
public:
    TranscendentalProbe();
    ~TranscendentalProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kTranscendental; }
    [[nodiscard]] std::string_view name() const noexcept override { return "transcendental"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 120; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kMedium; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kAll; }

    /// Number of input values per transcendental function.
    static constexpr uint32_t kValuesPerFunction = 256;

    /// Number of functions tested (sin, cos, exp, log, rsqrt).
    static constexpr uint32_t kNumFunctions = 5;

    /// Total output values.
    static constexpr uint32_t kTotalOutputs = kValuesPerFunction * kNumFunctions;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
