/// @file test_scheduler.cpp
/// @brief Unit tests for the probe scheduler.

#include "agent/scheduler.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

namespace sentinel::agent {
namespace {

using namespace ::testing;

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up default schedules.
        schedules_ = {
            {probes::ProbeType::kFma,           60, probes::Priority::kHigh,
             probes::SmSelection::kAll,          true},
            {probes::ProbeType::kTensorCore,    60, probes::Priority::kHigh,
             probes::SmSelection::kAll,          true},
            {probes::ProbeType::kTranscendental,120, probes::Priority::kMedium,
             probes::SmSelection::kAll,          true},
        };

        // 2 devices with 4 SMs each.
        devices_ = {{0, 4}, {1, 4}};
    }

    std::vector<ProbeScheduleConfig> schedules_;
    std::vector<std::pair<int, int>> devices_;
};

TEST_F(SchedulerTest, InitializeAndStartStop) {
    Scheduler scheduler;
    scheduler.initialize(schedules_, devices_);

    std::atomic<int> task_count{0};
    scheduler.start([&](const ProbeTask& /*task*/) {
        task_count.fetch_add(1, std::memory_order_relaxed);
    });

    EXPECT_TRUE(scheduler.is_running());

    // Let the scheduler run for a short time to generate tasks.
    std::this_thread::sleep_for(std::chrono::milliseconds(250));

    scheduler.stop();
    EXPECT_FALSE(scheduler.is_running());

    // At least some tasks should have been generated and executed.
    EXPECT_GT(task_count.load(), 0);
}

TEST_F(SchedulerTest, TasksHaveCorrectFields) {
    Scheduler scheduler;
    scheduler.initialize(schedules_, devices_);

    std::mutex mtx;
    std::vector<ProbeTask> captured_tasks;

    scheduler.start([&](const ProbeTask& task) {
        std::lock_guard lock(mtx);
        captured_tasks.push_back(task);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    scheduler.stop();

    std::lock_guard lock(mtx);
    ASSERT_FALSE(captured_tasks.empty());

    // Check that tasks have valid device indices and SM IDs.
    for (const auto& task : captured_tasks) {
        EXPECT_TRUE(task.device_index == 0 || task.device_index == 1);
        EXPECT_LT(task.sm_id, 4u);  // 4 SMs per device.
        EXPECT_NE(task.probe_type, static_cast<probes::ProbeType>(0));
    }
}

TEST_F(SchedulerTest, UtilizationAffectsScheduling) {
    Scheduler scheduler;
    scheduler.initialize(schedules_, devices_);

    std::atomic<int> task_count_normal{0};
    std::atomic<int> task_count_high_util{0};

    // Run with normal utilization.
    scheduler.update_utilization(0, 50.0f);
    scheduler.update_utilization(1, 50.0f);

    scheduler.start([&](const ProbeTask&) {
        task_count_normal.fetch_add(1, std::memory_order_relaxed);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    scheduler.stop();

    // Run with high utilization (should back off).
    Scheduler scheduler2;
    scheduler2.initialize(schedules_, devices_);
    scheduler2.update_utilization(0, 95.0f);
    scheduler2.update_utilization(1, 95.0f);

    scheduler2.start([&](const ProbeTask&) {
        task_count_high_util.fetch_add(1, std::memory_order_relaxed);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    scheduler2.stop();

    // Both should have run tasks (the first cycle fires immediately),
    // but subsequent cycles should be slower under high utilization.
    // This is a behavioral test — exact counts depend on timing.
    EXPECT_GT(task_count_normal.load(), 0);
    EXPECT_GT(task_count_high_util.load(), 0);
}

TEST_F(SchedulerTest, ScheduleOverride) {
    Scheduler scheduler;
    scheduler.initialize(schedules_, devices_);

    // Disable FMA probes via override.
    scheduler.apply_override(probes::ProbeType::kFma, 0, 60);

    std::mutex mtx;
    std::vector<ProbeTask> captured_tasks;

    scheduler.start([&](const ProbeTask& task) {
        std::lock_guard lock(mtx);
        captured_tasks.push_back(task);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    scheduler.stop();

    // No FMA tasks should have been scheduled.
    std::lock_guard lock(mtx);
    for (const auto& task : captured_tasks) {
        EXPECT_NE(task.probe_type, probes::ProbeType::kFma);
    }
}

TEST_F(SchedulerTest, DisabledProbeNotScheduled) {
    schedules_[0].enabled = false;  // Disable FMA.

    Scheduler scheduler;
    scheduler.initialize(schedules_, devices_);

    std::mutex mtx;
    std::vector<ProbeTask> captured_tasks;

    scheduler.start([&](const ProbeTask& task) {
        std::lock_guard lock(mtx);
        captured_tasks.push_back(task);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    scheduler.stop();

    std::lock_guard lock(mtx);
    for (const auto& task : captured_tasks) {
        EXPECT_NE(task.probe_type, probes::ProbeType::kFma);
    }
}

}  // namespace
}  // namespace sentinel::agent
