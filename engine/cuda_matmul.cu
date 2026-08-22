#include "engine/cuda_internal.h"

namespace {

__global__ void fp32_to_fp16(const float* source, __half* destination,
                             size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        destination[index] = __float2half(source[index]);
}

__global__ void dequantize_q8_dense_weight_cuda(
    const int8_t* weight, const float* scales, int group_size,
    int groups_per_row, __half* output, size_t count, int width) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index % width);
    const int group = min(column / group_size, groups_per_row - 1);
    output[index] = __float2half(
        static_cast<float>(weight[index]) *
        scales[static_cast<size_t>(row) * groups_per_row + group]);
}

__global__ void q8_dense_gemv_cuda(
    const float* activation, const int8_t* weight, const float* scales,
    int group_size, int groups_per_row, float* output, int columns,
    int inner) {
    constexpr int warps_per_block = 4;
    const int warp = static_cast<int>(threadIdx.x) / warpSize;
    const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
    const int column = static_cast<int>(blockIdx.x) * warps_per_block + warp;
    if (column >= columns)
        return;
    float sum = 0.0f;
    const int8_t* row = weight + static_cast<size_t>(column) * inner;
    const float* row_scales =
        scales + static_cast<size_t>(column) * groups_per_row;
    for (int group = 0; group < groups_per_row; ++group) {
        const int begin = group * group_size;
        const int end = min(inner, begin + group_size);
        const float scale = row_scales[group];
        for (int k = begin + lane; k < end; k += warpSize)
            sum += activation[k] * static_cast<float>(row[k]) * scale;
    }
    for (int offset = warpSize / 2; offset != 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0)
        output[column] = sum;
}

}  // namespace

namespace mollm_cuda {

void launch_fp32_to_fp16(const float* source, __half* destination,
                         size_t count) {
    constexpr int threads = 256;
    fp32_to_fp16<<<static_cast<unsigned>((count + threads - 1) / threads),
                   threads>>>(source, destination, count);
}

void launch_dequantize_q8_dense_weight(
    const int8_t* weight, const float* scales, int group_size,
    int groups_per_row, __half* output, size_t count, int width) {
    constexpr int threads = 256;
    dequantize_q8_dense_weight_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        weight, scales, group_size, groups_per_row, output, count, width);
}

void launch_q8_dense_gemv(
    const float* activation, const int8_t* weight, const float* scales,
    int group_size, int groups_per_row, float* output, int columns,
    int inner) {
    constexpr int warps_per_block = 4;
    constexpr int threads = warps_per_block * 32;
    const unsigned blocks = static_cast<unsigned>(
        (columns + warps_per_block - 1) / warps_per_block);
    q8_dense_gemv_cuda<<<blocks, threads>>>(
        activation, weight, scales, group_size, groups_per_row, output,
        columns, inner);
}

bool run_dense_matmul(
    cublasHandle_t cublas, const void* weight, cudaDataType weight_type,
    const void* activation, cudaDataType activation_type, float* output,
    int m, int n, int k, int lda) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C[M,N] = A[M,K] * W[N,K]^T is the equivalent
    // column-major operation C_col[N,M] = W_col[K,N]^T * A_col[K,M].
    return report_cublas(
        cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, weight,
            weight_type, k, activation, activation_type, lda, &beta, output,
            CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        "cublasGemmEx");
}

}  // namespace mollm_cuda
