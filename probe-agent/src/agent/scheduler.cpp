/// @file scheduler.cpp
/// @brief Probe scheduler implementation.

#include "agent/scheduler.h"
#include "probes/sm_affinity.h"
#include "util/logging.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <unordered_map>

namespace sentinel::agent {

// ── Internal types ──────────────────────────────────────────────────

struct PerProbeState {
    ProbeScheduleConfig config;
    std::chrono::steady_clock::time_point next_fire;
    bool override_active = false;
    std::chrono::steady_clock::time_point override_expires;
    uint32_t original_period = 0;
};

struct PerDeviceState {
    int device_index = -1;
    int sm_count = 0;
    float utilization = 0.0f;
};

/// Priority comparison: higher priority and earlier deadline first.
struct TaskCompare {
    bool operator()(const ProbeTask& a, const ProbeTask& b) const {
        if (a.priority != b.priority) {
            return a.priority < b.priority;  // Higher priority first.
        }
        return a.scheduled_time > b.scheduled_time;  // Earlier time first.
    }
};

// ── Impl ─────────────────────────────────────────────────────────────

struct Scheduler::Impl {
    std::atomic<bool> running{false};
    std::thread scheduler_thread;

    mutable std::mutex mutex;
    std::condition_variable cv;

    std::vector<PerProbeState> probe_states;
    std::vector<PerDeviceState> device_states;

    std::priority_queue<ProbeTask, std::vector<ProbeTask>, TaskCompare> task_queue;
    ProbeExecutor executor;

    std::mt19937 rng{std::random_device{}()};

    /// Compute the schedule multiplier based on GPU utilization.
    /// Above backoff threshold (90%): slow down by 4x.
    /// Below burst threshold (10%): speed up by 2x.
    /// Between: normal (1x).
    float utilization_multiplier(float utilization) const {
        if (utilization >= 90.0f) return 4.0f;
        if (utilization >= 75.0f) return 2.0f;
        if (utilization <= 10.0f) return 0.5f;
        return 1.0f;
    }

    /// Select which SMs to test based on the selection strategy.
    std::vector<uint32_t> select_sms(const PerDeviceState& dev,
                                      probes::SmSelection sel) {
        std::vector<uint32_t> sms;
        switch (sel) {
            case probes::SmSelection::kAll:
                sms.resize(dev.sm_count);
                std::iota(sms.begin(), sms.end(), 0u);
                break;
            case probes::SmSelection::kSample25Pct: {
                int count = 0;
                sms.resize(dev.sm_count);
                probes::select_random_sms(dev.sm_count, 0.25f,
                                           sms.data(), count,
                                           static_cast<int>(sms.size()));
                sms.resize(count);
                break;
            }
            case probes::SmSelection::kSample10Pct: {
                int count = 0;
                sms.resize(dev.sm_count);
                probes::select_random_sms(dev.sm_count, 0.10f,
                                           sms.data(), count,
                                           static_cast<int>(sms.size()));
                sms.resize(count);
                break;
            }
        }
        return sms;
    }

    /// Generate probe tasks for the next scheduling cycle.
    void generate_tasks() {
        auto now = std::chrono::steady_clock::now();

        for (auto& ps : probe_states) {
            if (!ps.config.enabled) continue;
            if (now < ps.next_fire) continue;

            // Check and revert expired overrides.
            if (ps.override_active && now >= ps.override_expires) {
                ps.config.period_seconds = ps.original_period;
                ps.override_active = false;
                SENTINEL_LOG_INFO("Schedule override expired for {}",
                                  probes::probe_type_name(ps.config.probe_type));
            }

            // Generate tasks for each device.
            for (auto& dev : device_states) {
                float mult = utilization_multiplier(dev.utilization);
                auto sms = select_sms(dev, ps.config.sm_selection);

                for (uint32_t sm_id : sms) {
                    ProbeTask task;
                    task.probe_type = ps.config.probe_type;
                    task.sm_id = sm_id;
                    task.device_index = dev.device_index;
                    task.priority = ps.config.priority;
                    task.scheduled_time = now;
                    task.deadline = now + std::chrono::seconds(
                        static_cast<int>(ps.config.period_seconds * mult));
                    task_queue.push(task);
                }
            }

            // Compute next fire time with utilization adjustment.
            float avg_util = 0.0f;
            for (const auto& dev : device_states) {
                avg_util += dev.utilization;
            }
            if (!device_states.empty()) {
                avg_util /= static_cast<float>(device_states.size());
            }
            float mult = utilization_multiplier(avg_util);
            auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<float>(ps.config.period_seconds * mult));
            ps.next_fire = now + period;
        }
    }

    /// Main scheduler loop.
    void run() {
        SENTINEL_LOG_INFO("Scheduler thread started");
        while (running.load(std::memory_order_relaxed)) {
            {
                std::unique_lock lock(mutex);

                generate_tasks();

                // Dispatch all ready tasks.
                while (!task_queue.empty()) {
                    ProbeTask task = task_queue.top();
                    task_queue.pop();

                    // Release lock while executing to avoid blocking.
                    lock.unlock();
                    try {
                        executor(task);
                    } catch (const std::exception& e) {
                        SENTINEL_LOG_ERROR("Probe execution failed: {}", e.what());
                    }
                    lock.lock();
                }
            }

            // Sleep until next scheduling cycle (100ms polling interval).
            std::unique_lock lock(mutex);
            cv.wait_for(lock, std::chrono::milliseconds(100),
                         [this] { return !running.load(std::memory_order_relaxed); });
        }
        SENTINEL_LOG_INFO("Scheduler thread stopped");
    }
};

// ── Public interface ────────────────────────────────────────────────

Scheduler::Scheduler() : impl_(std::make_unique<Impl>()) {}
Scheduler::~Scheduler() { stop(); }

void Scheduler::initialize(const std::vector<ProbeScheduleConfig>& schedules,
                             const std::vector<std::pair<int, int>>& devices) {
    std::lock_guard lock(impl_->mutex);

    auto now = std::chrono::steady_clock::now();

    impl_->probe_states.clear();
    for (const auto& sched : schedules) {
        PerProbeState ps;
        ps.config = sched;
        ps.next_fire = now;
        ps.original_period = sched.period_seconds;
        impl_->probe_states.push_back(ps);
    }

    impl_->device_states.clear();
    for (const auto& [dev_idx, sm_count] : devices) {
        PerDeviceState ds;
        ds.device_index = dev_idx;
        ds.sm_count = sm_count;
        impl_->device_states.push_back(ds);
    }

    SENTINEL_LOG_INFO("Scheduler initialized with {} probe types across {} devices",
                      schedules.size(), devices.size());
}

void Scheduler::start(ProbeExecutor executor) {
    if (impl_->running.load()) return;

    impl_->executor = std::move(executor);
    impl_->running.store(true);
    impl_->scheduler_thread = std::thread(&Impl::run, impl_.get());
}

void Scheduler::stop() {
    if (!impl_->running.load()) return;

    impl_->running.store(false);
    impl_->cv.notify_all();

    if (impl_->scheduler_thread.joinable()) {
        impl_->scheduler_thread.join();
    }
}

void Scheduler::update_utilization(int device_index, float utilization) {
    std::lock_guard lock(impl_->mutex);
    for (auto& dev : impl_->device_states) {
        if (dev.device_index == device_index) {
            dev.utilization = utilization;
            break;
        }
    }
}

void Scheduler::apply_override(probes::ProbeType probe_type,
                                uint32_t period_seconds,
                                uint32_t duration_seconds) {
    std::lock_guard lock(impl_->mutex);
    for (auto& ps : impl_->probe_states) {
        if (ps.config.probe_type == probe_type) {
            ps.original_period = ps.config.period_seconds;
            ps.config.period_seconds = period_seconds;
            ps.config.enabled = (period_seconds > 0);
            ps.override_active = true;
            ps.override_expires = std::chrono::steady_clock::now() +
                                  std::chrono::seconds(duration_seconds);
            SENTINEL_LOG_INFO("Schedule override applied for {}: period={}s for {}s",
                              probes::probe_type_name(probe_type),
                              period_seconds, duration_seconds);
            break;
        }
    }
}

void Scheduler::update_schedules(const std::vector<ProbeScheduleConfig>& schedules) {
    std::lock_guard lock(impl_->mutex);
    for (const auto& sched : schedules) {
        bool found = false;
        for (auto& ps : impl_->probe_states) {
            if (ps.config.probe_type == sched.probe_type) {
                if (!ps.override_active) {
                    ps.config = sched;
                    ps.original_period = sched.period_seconds;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            PerProbeState ps;
            ps.config = sched;
            ps.next_fire = std::chrono::steady_clock::now();
            ps.original_period = sched.period_seconds;
            impl_->probe_states.push_back(ps);
        }
    }
}

std::size_t Scheduler::pending_task_count() const {
    std::lock_guard lock(impl_->mutex);
    return impl_->task_queue.size();
}

bool Scheduler::is_running() const {
    return impl_->running.load();
}

}  // namespace sentinel::agent
