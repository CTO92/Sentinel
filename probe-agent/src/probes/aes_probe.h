/// @file aes_probe.h
/// @brief AES-128-ECB combinational logic probe.
///
/// Encrypts a 4KB known plaintext using a fixed AES-128 key entirely
/// on-GPU. Bit-exact comparison against golden ciphertext detects
/// stuck-at faults or SDC in the integer/logic datapaths.

#pragma once

#include "probes/probe_interface.h"

#include <memory>

namespace sentinel::probes {

class AesProbe : public ProbeInterface {
public:
    AesProbe();
    ~AesProbe() override;

    [[nodiscard]] ProbeType type() const noexcept override { return ProbeType::kAes; }
    [[nodiscard]] std::string_view name() const noexcept override { return "aes"; }

    bool initialize(int device_index) override;
    [[nodiscard]] ProbeResult execute(uint32_t sm_id, cudaStream_t stream) override;
    void teardown() override;

    [[nodiscard]] std::size_t memory_footprint() const noexcept override;
    [[nodiscard]] uint32_t default_period_seconds() const noexcept override { return 300; }
    [[nodiscard]] Priority default_priority() const noexcept override { return Priority::kMedium; }
    [[nodiscard]] SmSelection default_sm_selection() const noexcept override { return SmSelection::kSample25Pct; }

    /// Size of the plaintext/ciphertext buffer in bytes.
    static constexpr uint32_t kDataSize = 4096;

    /// AES-128 block size in bytes.
    static constexpr uint32_t kBlockSize = 16;

    /// Number of AES-128 blocks in the data.
    static constexpr uint32_t kNumBlocks = kDataSize / kBlockSize;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::probes
