/// @file fma_probe.h
/// @brief Fused Multiply-Add (FMA) determinism probe.
///
/// Tests FP32 FMA units on individual SMs by executing a*b+c on 1024
/// carefully chosen test vectors covering:
///   - Normal range (small, medium, large exponents)
///   - Denormals (gradual underflow)
///   - Exponent boundaries (near overflow/underflow)
///   - Mantissa patterns (all-ones, alternating, single-bit)
///
/// Each vector is executed 16 times to detect intermittent errors.
/// Results are compared bit-exact against precomputed golden answers.

#pragma once

#include "probes/probe_interface.h"

#include <cstdint>
#include <memory>

namespace sentinel::probes {

class FmaProbe : public ProbeInterface {
public:
    FmaProbe();
    ~FmaProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kFma; }
    [[nodiscard]] std::string_view name() const noexcept override { return "fma"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 60; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kHigh; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kAll; }

    /// Number of FMA test vectors.
    static constexpr uint32_t kNumVectors = 1024;

    /// Number of repetitions per vector.
    static constexpr uint32_t kRepetitions = 16;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
