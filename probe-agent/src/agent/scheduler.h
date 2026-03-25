/// @file scheduler.h
/// @brief Probe execution scheduler.
///
/// Maintains a priority queue of probe tasks based on their configured
/// periods, dynamically adjusts scheduling based on GPU utilization,
/// and dispatches work to executor thread pools. Handles SM selection
/// strategies and schedule overrides from the Correlation Engine.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "agent/config_manager.h"
#include "probes/probe_interface.h"

namespace sentinel::agent {

/// A scheduled probe task ready for execution.
struct ProbeTask {
    probes::ProbeType probe_type;
    uint32_t sm_id;
    int device_index;
    probes::Priority priority;
    std::chrono::steady_clock::time_point scheduled_time;
    std::chrono::steady_clock::time_point deadline;
};

/// Callback invoked by the scheduler to execute a probe task.
using ProbeExecutor = std::function<void(const ProbeTask&)>;

/// Probe execution scheduler.
class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// Initialize the scheduler with probe configuration and device info.
    /// @param schedules   Per-probe-type scheduling configs.
    /// @param devices     List of (device_index, sm_count) pairs.
    void initialize(const std::vector<ProbeScheduleConfig>& schedules,
                     const std::vector<std::pair<int, int>>& devices);

    /// Start the scheduler thread.
    /// @param executor Callback invoked for each scheduled task.
    void start(ProbeExecutor executor);

    /// Stop the scheduler thread and wait for it to finish.
    void stop();

    /// Update GPU utilization for dynamic scheduling.
    /// @param device_index GPU device index.
    /// @param utilization  Utilization percentage (0-100).
    void update_utilization(int device_index, float utilization);

    /// Apply a schedule override for a specific probe type.
    void apply_override(probes::ProbeType probe_type,
                         uint32_t period_seconds,
                         uint32_t duration_seconds);

    /// Update schedules from a new config.
    void update_schedules(const std::vector<ProbeScheduleConfig>& schedules);

    /// @return Number of tasks currently pending in the queue.
    [[nodiscard]] std::size_t pending_task_count() const;

    /// @return true if the scheduler is running.
    [[nodiscard]] bool is_running() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace sentinel::agent
