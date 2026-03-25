/// @file tensor_core_probe.h
/// @brief Tensor Core matrix-multiply reproducibility probe.
///
/// Tests Tensor Core units by performing 16x16 matrix operations with
/// known inputs (identity, permutation, Hadamard matrices) in both
/// FP16 and INT8 precision. Results must be bit-exact against golden
/// values — zero tolerance for SDC.

#pragma once

#include "probes/probe_interface.h"

#include <memory>

namespace sentinel::probes {

class TensorCoreProbe : public ProbeInterface {
public:
    TensorCoreProbe();
    ~TensorCoreProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kTensorCore; }
    [[nodiscard]] std::string_view name() const noexcept override { return "tensor_core"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 60; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kHigh; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kAll; }

    /// Matrix dimension for tensor core tests (16x16 tiles).
    static constexpr int kMatDim = 16;

    /// Number of matrix test cases.
    static constexpr int kNumTestCases = 6;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
