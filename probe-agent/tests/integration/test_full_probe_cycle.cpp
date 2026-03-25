/// @file test_full_probe_cycle.cpp
/// @brief Integration test: full probe execution cycle with all probe types.
///
/// Requires a CUDA GPU. Initializes all probes on GPU 0, executes each
/// on SM 0, and verifies pass results. This validates the complete
/// probe pipeline from initialization through execution and result
/// collection.

#include "probes/fma_probe.h"
#include "probes/tensor_core_probe.h"
#include "probes/transcendental_probe.h"
#include "probes/aes_probe.h"
#include "probes/memory_probe.h"
#include "probes/register_file_probe.h"
#include "probes/shared_memory_probe.h"
#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"
#include "util/crypto.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

namespace sentinel::probes {
namespace {

class FullProbeCycleTest : public ::testing::Test {
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

        platform::set_device(device_index_);
        stream_ = std::make_unique<platform::CudaStream>();
    }

    void TearDown() override {
        stream_.reset();
    }

    int device_index_ = 0;
    int sm_count_ = 0;
    std::unique_ptr<platform::CudaStream> stream_;
};

TEST_F(FullProbeCycleTest, FmaProbeFullCycle) {
    FmaProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult result = probe.execute(0, stream_->get());
    EXPECT_TRUE(result.match);
    EXPECT_EQ(result.probe_type, ProbeType::kFma);
    EXPECT_GT(result.execution_time_ns, 0u);
    EXPECT_FALSE(util::digest_to_hex(result.expected_hash).empty());

    probe.teardown();
}

TEST_F(FullProbeCycleTest, TensorCoreProbeFullCycle) {
    // Check compute capability for tensor core support.
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index_));
    if (prop.major < 7) {
        GTEST_SKIP() << "Tensor Cores require compute capability 7.0+";
    }

    TensorCoreProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult result = probe.execute(0, stream_->get());
    EXPECT_TRUE(result.match);
    EXPECT_EQ(result.probe_type, ProbeType::kTensorCore);
    EXPECT_GT(result.execution_time_ns, 0u);

    probe.teardown();
}

TEST_F(FullProbeCycleTest, TranscendentalProbeFullCycle) {
    TranscendentalProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult result = probe.execute(0, stream_->get());
    EXPECT_TRUE(result.match);
    EXPECT_EQ(result.probe_type, ProbeType::kTranscendental);
    EXPECT_GT(result.execution_time_ns, 0u);

    probe.teardown();
}

TEST_F(FullProbeCycleTest, AesProbeFullCycle) {
    AesProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult result = probe.execute(0, stream_->get());
    EXPECT_TRUE(result.match);
    EXPECT_EQ(result.probe_type, ProbeType::kAes);
    EXPECT_GT(result.execution_time_ns, 0u);

    probe.teardown();
}

TEST_F(FullProbeCycleTest, MemoryProbeFullCycle) {
    MemoryProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult result = probe.execute(0, stream_->get());
    EXPECT_TRUE(result.match);
    EXPECT_EQ(result.probe_type, ProbeType::kMemory);
    EXPECT_GT(result.execution_time_ns, 0u);

    probe.teardown();
}

TEST_F(FullProbeCycleTest, RegisterFileProbeFullCycle) {
    RegisterFileProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult result = probe.execute(0, stream_->get());
    EXPECT_TRUE(result.match);
    EXPECT_EQ(result.probe_type, ProbeType::kRegisterFile);
    EXPECT_GT(result.execution_time_ns, 0u);

    probe.teardown();
}

TEST_F(FullProbeCycleTest, SharedMemoryProbeFullCycle) {
    SharedMemoryProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult result = probe.execute(0, stream_->get());
    EXPECT_TRUE(result.match);
    EXPECT_EQ(result.probe_type, ProbeType::kSharedMemory);
    EXPECT_GT(result.execution_time_ns, 0u);

    probe.teardown();
}

TEST_F(FullProbeCycleTest, AllProbesSequential) {
    // Run all probes in sequence on SM 0 and collect results.
    struct TestCase {
        std::unique_ptr<ProbeInterface> probe;
        bool requires_tensor_cores;
    };

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_index_));
    bool has_tensor_cores = (prop.major >= 7);

    std::vector<TestCase> cases;
    cases.push_back({std::make_unique<FmaProbe>(), false});
    cases.push_back({std::make_unique<TensorCoreProbe>(), true});
    cases.push_back({std::make_unique<TranscendentalProbe>(), false});
    cases.push_back({std::make_unique<AesProbe>(), false});
    cases.push_back({std::make_unique<MemoryProbe>(), false});
    cases.push_back({std::make_unique<RegisterFileProbe>(), false});
    cases.push_back({std::make_unique<SharedMemoryProbe>(), false});

    int pass_count = 0;
    int skip_count = 0;

    for (auto& tc : cases) {
        if (tc.requires_tensor_cores && !has_tensor_cores) {
            ++skip_count;
            continue;
        }

        ASSERT_TRUE(tc.probe->initialize(device_index_))
            << "Failed to initialize " << tc.probe->name();

        ProbeResult result = tc.probe->execute(0, stream_->get());

        EXPECT_TRUE(result.match)
            << tc.probe->name() << " probe failed on SM 0";

        if (result.match) ++pass_count;

        tc.probe->teardown();
    }

    int expected = static_cast<int>(cases.size()) - skip_count;
    EXPECT_EQ(pass_count, expected)
        << pass_count << "/" << expected << " probes passed";
}

TEST_F(FullProbeCycleTest, ProbeResultHashConsistency) {
    // Verify that running the same probe twice produces the same hash.
    FmaProbe probe;
    ASSERT_TRUE(probe.initialize(device_index_));

    ProbeResult r1 = probe.execute(0, stream_->get());
    ProbeResult r2 = probe.execute(0, stream_->get());

    EXPECT_EQ(r1.expected_hash, r2.expected_hash);
    EXPECT_EQ(r1.actual_hash, r2.actual_hash);
    EXPECT_TRUE(r1.match);
    EXPECT_TRUE(r2.match);

    probe.teardown();
}

TEST_F(FullProbeCycleTest, MemoryFootprintReasonable) {
    // Verify memory footprint estimates are in a reasonable range.
    FmaProbe fma;
    EXPECT_GT(fma.memory_footprint(), 0u);
    EXPECT_LT(fma.memory_footprint(), 1024 * 1024 * 10);  // < 10 MB.

    MemoryProbe mem;
    EXPECT_GT(mem.memory_footprint(), 16 * 1024 * 1024);  // >= 16 MB test region.
    EXPECT_LT(mem.memory_footprint(), 32 * 1024 * 1024);  // < 32 MB total.
}

}  // namespace
}  // namespace sentinel::probes
