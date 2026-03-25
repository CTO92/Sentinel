/// @file sm_affinity.h
/// @brief SM-pinning primitives for targeting specific streaming multiprocessors.
///
/// Uses persistent-style kernels with SM ID readback via inline PTX
/// to ensure probe kernels execute on a designated SM. Thread blocks
/// read their SM ID and only perform work if it matches the target.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace sentinel::probes {

/// Read the SM ID of the current thread block via inline PTX.
/// This is a device-side function — call only from __device__ or __global__ code.
__device__ inline uint32_t read_sm_id() {
    uint32_t sm_id;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
    return sm_id;
}

/// Compute the recommended grid dimensions to ensure at least one block
/// lands on the target SM. We over-launch blocks so that at least one
/// block is scheduled on every SM; blocks on non-target SMs exit early.
///
/// @param sm_count          Total SMs on the device.
/// @param threads_per_block Threads per block for the probe kernel.
/// @param[out] grid_dim     Output grid dimension (number of blocks).
/// @param[out] block_dim    Output block dimension.
void compute_sm_pinned_launch_params(int sm_count,
                                      int threads_per_block,
                                      dim3& grid_dim,
                                      dim3& block_dim);

/// Query the number of SMs on the specified CUDA device.
///
/// @param device_index CUDA device index.
/// @return Number of streaming multiprocessors, or 0 on error.
[[nodiscard]] int get_sm_count(int device_index);

/// Select a random subset of SM IDs.
///
/// @param sm_count       Total number of SMs.
/// @param fraction       Fraction to select (e.g., 0.25 for 25%).
/// @param[out] selected  Output vector of selected SM IDs.
/// @param[out] count     Number of selected SMs.
/// @param max_output     Maximum entries in `selected`.
void select_random_sms(int sm_count, float fraction,
                        uint32_t* selected, int& count, int max_output);

}  // namespace sentinel::probes
