/// @file sm_affinity.cu
/// @brief SM-pinning implementation: launch parameter computation and SM selection.

#include "probes/sm_affinity.h"
#include "probes/probe_interface.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_runtime.h>

namespace sentinel::probes {

void compute_sm_pinned_launch_params(int sm_count,
                                      int threads_per_block,
                                      dim3& grid_dim,
                                      dim3& block_dim) {
    // Launch enough blocks so that every SM gets at least one block.
    // The CUDA scheduler distributes blocks round-robin across SMs,
    // so launching sm_count * 2 blocks ensures each SM gets at least
    // one (with high probability). Blocks that land on a non-target
    // SM exit immediately after checking read_sm_id().
    //
    // For persistent-style pinning, we use exactly sm_count * 2 blocks
    // with low occupancy per block. Each block runs threads_per_block
    // threads.
    int blocks = sm_count * 2;
    grid_dim = dim3(static_cast<unsigned>(blocks), 1, 1);
    block_dim = dim3(static_cast<unsigned>(threads_per_block), 1, 1);
}

int get_sm_count(int device_index) {
    cudaDeviceProp prop{};
    cudaError_t err = cudaGetDeviceProperties(&prop, device_index);
    if (err != cudaSuccess) {
        return 0;
    }
    return prop.multiProcessorCount;
}

void select_random_sms(int sm_count, float fraction,
                        uint32_t* selected, int& count, int max_output) {
    if (sm_count <= 0 || fraction <= 0.0f) {
        count = 0;
        return;
    }

    // Clamp fraction to [0, 1].
    fraction = std::min(1.0f, std::max(0.0f, fraction));

    int desired = static_cast<int>(std::ceil(static_cast<float>(sm_count) * fraction));
    desired = std::min(desired, max_output);
    desired = std::min(desired, sm_count);

    // Build a shuffled list of SM indices and take the first `desired`.
    std::vector<uint32_t> indices(static_cast<std::size_t>(sm_count));
    std::iota(indices.begin(), indices.end(), 0u);

    // Use a thread-local random engine seeded from hardware entropy.
    thread_local std::mt19937 rng{std::random_device{}()};
    std::shuffle(indices.begin(), indices.end(), rng);

    count = desired;
    std::copy_n(indices.begin(), desired, selected);
}

// ── ProbeInterface static helpers ─────────────────────────────────────

uint64_t ProbeInterface::next_probe_id() {
    static std::atomic<uint64_t> counter{0};
    return counter.fetch_add(1, std::memory_order_relaxed);
}

std::vector<std::unique_ptr<ProbeInterface>> create_all_probes() {
    // Deferred include to avoid circular dependencies — implementations
    // are linked from their respective .cu translation units.
    std::vector<std::unique_ptr<ProbeInterface>> probes;
    // Probe headers are included by the callers; here we just need the
    // forward-declared types. The actual construction happens via the
    // agent code that includes all probe headers.
    return probes;
}

}  // namespace sentinel::probes
