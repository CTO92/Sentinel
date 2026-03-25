/// @file test_fma_probe.cpp
/// @brief Unit tests for the FMA probe (requires CUDA GPU).

#include "probes/fma_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"

#include <gtest/gtest.h>

namespace sentinel::probes {
namespace {

class FmaProbeTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA device available; skipping GPU tests";
        }
        device_index_ = 0;
        sm_count_ = get_sm_count(device_index_);
        ASSERT_GT(sm_count_, 0);
    }

    int device_index_ = 0;
    int sm_count_ = 0;
};

TEST_F(FmaProbeTest, InitializeSucceeds) {
    FmaProbe probe;
    EXPECT_TRUE(probe.initialize(device_index_));
    EXPECT_EQ(probe.type(), ProbeType::kFma);
    EXPECT_EQ(probe.name(), "fma");
    EXPECT_GT(probe.memory_footprint(), 0u);
    probe.teardown();
}

TEST_F(FmaProbeTest, ExecuteOnSm0Passes) {
    FmaProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    platform::set_device(device_index_);
    platform::CudaStream stream;

    ProbeResult result = probe.execute(0, stream.get());
    EXPECT_EQ(result.probe_type, ProbeType::kFma);
    EXPECT_EQ(result.sm_id, 0u);

    // On healthy hardware, FMA should pass.
    EXPECT_TRUE(result.match) << "FMA probe failed on SM 0 — possible SDC or "
                                 "test environment issue";
    EXPECT_TRUE(result.mismatch_details.empty());
    EXPECT_GT(result.execution_time_ns, 0u);

    probe.teardown();
}

TEST_F(FmaProbeTest, ExecuteOnAllSMs) {
    FmaProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    platform::set_device(device_index_);
    platform::CudaStream stream;

    int pass_count = 0;
    int fail_count = 0;

    for (int sm = 0; sm < sm_count_; ++sm) {
        ProbeResult result = probe.execute(static_cast<uint32_t>(sm), stream.get());
        if (result.match) {
            ++pass_count;
        } else {
            ++fail_count;
        }
    }

    EXPECT_EQ(fail_count, 0) << fail_count << " SMs failed the FMA probe";
    EXPECT_EQ(pass_count, sm_count_);

    probe.teardown();
}

TEST_F(FmaProbeTest, ProbeMetadata) {
    FmaProbe probe;
    EXPECT_EQ(probe.default_period_seconds(), 60u);
    EXPECT_EQ(probe.default_priority(), Priority::kHigh);
    EXPECT_EQ(probe.default_sm_selection(), SmSelection::kAll);
}

TEST_F(FmaProbeTest, RepeatedExecutionDeterministic) {
    FmaProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    platform::set_device(device_index_);
    platform::CudaStream stream;

    ProbeResult r1 = probe.execute(0, stream.get());
    ProbeResult r2 = probe.execute(0, stream.get());

    EXPECT_EQ(r1.actual_hash, r2.actual_hash)
        << "FMA probe produced different results on consecutive runs";
    EXPECT_TRUE(r1.match);
    EXPECT_TRUE(r2.match);

    probe.teardown();
}

}  // namespace
}  // namespace sentinel::probes
