/// @file test_sm_affinity.cpp
/// @brief Unit tests for SM affinity / pinning utilities (requires CUDA GPU).

#include "probes/sm_affinity.h"
#include "platform/cuda_runtime.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <set>
#include <vector>

namespace sentinel::probes {
namespace {

class SmAffinityTest : public ::testing::Test {
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

TEST_F(SmAffinityTest, GetSmCount) {
    int count = get_sm_count(device_index_);
    EXPECT_GT(count, 0);

    // Invalid device should return 0.
    EXPECT_EQ(get_sm_count(999), 0);
}

TEST_F(SmAffinityTest, ComputeLaunchParams) {
    dim3 grid, block;
    compute_sm_pinned_launch_params(sm_count_, 256, grid, block);

    EXPECT_EQ(block.x, 256u);
    EXPECT_EQ(block.y, 1u);
    EXPECT_EQ(block.z, 1u);
    EXPECT_EQ(grid.x, static_cast<unsigned>(sm_count_ * 2));
    EXPECT_EQ(grid.y, 1u);
    EXPECT_EQ(grid.z, 1u);
}

TEST_F(SmAffinityTest, SelectRandomSms_All) {
    std::vector<uint32_t> selected(sm_count_);
    int count = 0;

    select_random_sms(sm_count_, 1.0f, selected.data(), count,
                       static_cast<int>(selected.size()));

    EXPECT_EQ(count, sm_count_);

    // All SM IDs should be present (complete permutation).
    std::set<uint32_t> unique_ids(selected.begin(), selected.begin() + count);
    EXPECT_EQ(static_cast<int>(unique_ids.size()), sm_count_);
}

TEST_F(SmAffinityTest, SelectRandomSms_25Pct) {
    std::vector<uint32_t> selected(sm_count_);
    int count = 0;

    select_random_sms(sm_count_, 0.25f, selected.data(), count,
                       static_cast<int>(selected.size()));

    int expected_min = std::max(1, static_cast<int>(sm_count_ * 0.25f));
    EXPECT_GE(count, expected_min);
    EXPECT_LE(count, sm_count_);

    // All selected IDs should be valid.
    for (int i = 0; i < count; ++i) {
        EXPECT_LT(selected[i], static_cast<uint32_t>(sm_count_));
    }
}

TEST_F(SmAffinityTest, SelectRandomSms_10Pct) {
    std::vector<uint32_t> selected(sm_count_);
    int count = 0;

    select_random_sms(sm_count_, 0.10f, selected.data(), count,
                       static_cast<int>(selected.size()));

    EXPECT_GE(count, 1);  // At least 1 SM selected.
    EXPECT_LE(count, sm_count_);
}

TEST_F(SmAffinityTest, SelectRandomSms_ZeroFraction) {
    std::vector<uint32_t> selected(sm_count_);
    int count = 0;

    select_random_sms(sm_count_, 0.0f, selected.data(), count,
                       static_cast<int>(selected.size()));

    EXPECT_EQ(count, 0);
}

TEST_F(SmAffinityTest, SelectRandomSms_InvalidSmCount) {
    std::vector<uint32_t> selected(4);
    int count = 0;

    select_random_sms(0, 0.5f, selected.data(), count,
                       static_cast<int>(selected.size()));
    EXPECT_EQ(count, 0);

    select_random_sms(-1, 0.5f, selected.data(), count,
                       static_cast<int>(selected.size()));
    EXPECT_EQ(count, 0);
}

TEST_F(SmAffinityTest, SelectRandomSms_Randomness) {
    // Run selection multiple times and verify we get different orderings.
    std::vector<uint32_t> selected1(sm_count_);
    std::vector<uint32_t> selected2(sm_count_);
    int count1 = 0, count2 = 0;

    select_random_sms(sm_count_, 1.0f, selected1.data(), count1,
                       static_cast<int>(selected1.size()));
    select_random_sms(sm_count_, 1.0f, selected2.data(), count2,
                       static_cast<int>(selected2.size()));

    EXPECT_EQ(count1, sm_count_);
    EXPECT_EQ(count2, sm_count_);

    // With sm_count > 2, two random permutations are very unlikely to be identical.
    if (sm_count_ > 2) {
        bool same = std::equal(selected1.begin(), selected1.begin() + count1,
                               selected2.begin());
        // This is probabilistic; with 3+ SMs, collisions are 1/n!.
        // We do not assert, just note.
        if (same) {
            std::cerr << "Note: two random SM selections were identical "
                         "(extremely unlikely but possible)\n";
        }
    }
}

}  // namespace
}  // namespace sentinel::probes
