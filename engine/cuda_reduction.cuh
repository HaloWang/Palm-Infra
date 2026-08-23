#pragma once

#include <cuda_runtime.h>

namespace mollm_cuda::detail {

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        value = fmaxf(
            value, __shfl_down_sync(0xffffffffu, value, offset));
    return value;
}

// Preserve the original 256-thread tree's addition order through stride 32,
// then finish the first warp with shuffles. This keeps quantized-model parity
// stable while avoiding five full-block barriers.
__device__ __forceinline__ float block_reduce_sum_256(
    float value, float* reduction) {
    const int thread = static_cast<int>(threadIdx.x);
    reduction[thread] = value;
    __syncthreads();
    if (thread < 128)
        reduction[thread] += reduction[thread + 128];
    __syncthreads();
    if (thread < 64)
        reduction[thread] += reduction[thread + 64];
    __syncthreads();
    if (thread < 32) {
        value = reduction[thread] + reduction[thread + 32];
        value = warp_reduce_sum(value);
        if (thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    return reduction[0];
}

__device__ __forceinline__ float block_reduce_max_256(
    float value, float* reduction) {
    const int thread = static_cast<int>(threadIdx.x);
    reduction[thread] = value;
    __syncthreads();
    if (thread < 128)
        reduction[thread] = fmaxf(
            reduction[thread], reduction[thread + 128]);
    __syncthreads();
    if (thread < 64)
        reduction[thread] = fmaxf(
            reduction[thread], reduction[thread + 64]);
    __syncthreads();
    if (thread < 32) {
        value = fmaxf(reduction[thread], reduction[thread + 32]);
        value = warp_reduce_max(value);
        if (thread == 0)
            reduction[0] = value;
    }
    __syncthreads();
    return reduction[0];
}

}  // namespace mollm_cuda::detail
