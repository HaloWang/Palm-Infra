#include "engine/cuda_internal.h"

#include <cmath>

namespace {

__device__ float warp_reduce_sum(float value) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}

// Preserve the original 256-thread tree's addition order through stride 32,
// then finish the first warp with shuffles.  This removes five block barriers
// without perturbing the numerical path used by quantized-model parity tests.
__device__ float block_reduce_sum_256(float value, float* reduction) {
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

__device__ float cuda_activation(float value, int kind) {
    switch (kind) {
    case 1: return value / (1.0f + expf(-value));
    case 2: {
        const float inner = 0.7978845608f *
            (value + 0.044715f * value * value * value);
        return 0.5f * value * (1.0f + tanhf(inner));
    }
    case 3: return fmaxf(value, 0.0f);
    case 4: {
        const float positive = fmaxf(value, 0.0f);
        return positive * positive;
    }
    default: return value;
    }
}

__global__ void apply_activation_cuda(float* values, int rows, int columns,
                                      int kind, int begin, int end) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(rows) * columns;
    if (index >= count)
        return;
    const int column = static_cast<int>(index % columns);
    if (column >= begin && column < end)
        values[index] = cuda_activation(values[index], kind);
}

__global__ void binary_cuda(const float* lhs, const float* rhs, float* output,
                            size_t count, bool multiply) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = multiply ? lhs[index] * rhs[index]
                                 : lhs[index] + rhs[index];
}

__global__ void sigmoid_mul_cuda(const float* value, const float* gate,
                                 float* output, size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < count)
        output[index] = value[index] / (1.0f + expf(-gate[index]));
}

__global__ void sigmoid_mul_strided_cuda(
    const float* value, const float* gate, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2,
    size_t value_s0, size_t value_s1, size_t value_s2, size_t value_s3,
    size_t gate_s0, size_t gate_s1, size_t gate_s2, size_t gate_s3) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t remaining = index;
    const size_t i0 = remaining % static_cast<size_t>(d0);
    remaining /= static_cast<size_t>(d0);
    const size_t i1 = remaining % static_cast<size_t>(d1);
    remaining /= static_cast<size_t>(d1);
    const size_t i2 = remaining % static_cast<size_t>(d2);
    const size_t i3 = remaining / static_cast<size_t>(d2);
    const float v = value[i0 * value_s0 + i1 * value_s1 + i2 * value_s2 +
                          i3 * value_s3];
    const float g = gate[i0 * gate_s0 + i1 * gate_s1 + i2 * gate_s2 +
                         i3 * gate_s3];
    output[index] = v / (1.0f + expf(-g));
}

__global__ void unary_cuda(const float* input, float* output, size_t count,
                           int operation) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    const float value = input[index];
    switch (operation) {
    case 0: output[index] = value / (1.0f + expf(-value)); break;
    case 1: {
        const float inner = 0.7978845608f *
            (value + 0.044715f * value * value * value);
        output[index] = 0.5f * value * (1.0f + tanhf(inner));
        break;
    }
    case 2: output[index] = tanhf(value); break;
    case 3: output[index] = 1.0f / (1.0f + expf(-value)); break;
    case 4: output[index] = expf(value); break;
    case 5:
        output[index] = value > 20.0f ? value : log1pf(expf(value));
        break;
    }
}

__global__ void swiglu_cuda(const float* input, float* output,
                            size_t output_count, size_t half) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index < output_count) {
        const size_t row = index / half;
        const size_t column = index % half;
        const size_t base = row * half * 2;
        const float gate = input[base + column];
        output[index] = gate / (1.0f + expf(-gate)) *
            input[base + half + column];
    }
}

__global__ void rms_norm_cuda(const float* input, const float* weight,
                              float* output, int width, int rows,
                              float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.0f;
    const float* source = input + static_cast<size_t>(row) * width;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = source[column];
        sum += value * value;
    }
    __shared__ float reduction[256];
    const float block_sum = block_reduce_sum_256(sum, reduction);
    const float inverse = rsqrtf(block_sum / width + epsilon);
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        output[static_cast<size_t>(row) * width + column] =
            source[column] * inverse * weight[column];
}

__global__ void add_rms_norm_cuda(
    float* residual, const float* update, const float* weight, float* output,
    int width, int rows, size_t residual_row_stride,
    size_t update_row_stride, size_t output_row_stride, float epsilon) {
    const int row = blockIdx.x;
    if (row >= rows)
        return;
    float sum = 0.0f;
    float* residual_row = residual + static_cast<size_t>(row) *
        residual_row_stride;
    const float* update_row = update + static_cast<size_t>(row) *
        update_row_stride;
    for (int column = threadIdx.x; column < width; column += blockDim.x) {
        const float value = residual_row[column] + update_row[column];
        residual_row[column] = value;
        sum += value * value;
    }
    __shared__ float reduction[256];
    const float block_sum = block_reduce_sum_256(sum, reduction);
    const float inverse = rsqrtf(block_sum / width + epsilon);
    float* output_row = output + static_cast<size_t>(row) *
        output_row_stride;
    for (int column = threadIdx.x; column < width; column += blockDim.x)
        output_row[column] = residual_row[column] * inverse * weight[column];
}

__global__ void contiguous_cuda(
    const float* input, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2,
    size_t s0, size_t s1, size_t s2, size_t s3) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    if (index >= count)
        return;
    size_t remaining = index;
    const size_t i0 = remaining % static_cast<size_t>(d0);
    remaining /= static_cast<size_t>(d0);
    const size_t i1 = remaining % static_cast<size_t>(d1);
    remaining /= static_cast<size_t>(d1);
    const size_t i2 = remaining % static_cast<size_t>(d2);
    const size_t i3 = remaining / static_cast<size_t>(d2);
    output[index] = input[i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3];
}

}  // namespace

namespace mollm_cuda {

void launch_apply_activation(float* values, int rows, int columns, int kind,
                             int begin, int end) {
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(rows) * columns;
    apply_activation_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        values, rows, columns, kind, begin, end);
}

void launch_binary(const float* lhs, const float* rhs, float* output,
                   size_t count, bool multiply) {
    constexpr int threads = 256;
    binary_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                  threads>>>(lhs, rhs, output, count, multiply);
}

void launch_sigmoid_mul(const float* value, const float* gate, float* output,
                        size_t count) {
    constexpr int threads = 256;
    sigmoid_mul_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                       threads>>>(value, gate, output, count);
}

void launch_sigmoid_mul_strided(
    const float* value, const float* gate, float* output, size_t count,
    int64_t d0, int64_t d1, int64_t d2, size_t value_s0, size_t value_s1,
    size_t value_s2, size_t value_s3, size_t gate_s0, size_t gate_s1,
    size_t gate_s2, size_t gate_s3) {
    constexpr int threads = 256;
    sigmoid_mul_strided_cuda<<<
        static_cast<unsigned>((count + threads - 1) / threads), threads>>>(
        value, gate, output, count, d0, d1, d2, value_s0, value_s1, value_s2,
        value_s3, gate_s0, gate_s1, gate_s2, gate_s3);
}

void launch_unary(const float* input, float* output, size_t count,
                  int operation) {
    constexpr int threads = 256;
    unary_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                 threads>>>(input, output, count, operation);
}

void launch_swiglu(const float* input, float* output, size_t output_count,
                   size_t half) {
    constexpr int threads = 256;
    swiglu_cuda<<<
        static_cast<unsigned>((output_count + threads - 1) / threads),
        threads>>>(input, output, output_count, half);
}

void launch_rms_norm(const float* input, const float* weight, float* output,
                     int width, int rows, float epsilon) {
    constexpr int threads = 256;
    rms_norm_cuda<<<rows, threads>>>(
        input, weight, output, width, rows, epsilon);
}

void launch_add_rms_norm(
    float* residual, const float* update, const float* weight, float* output,
    int width, int rows, size_t residual_row_stride,
    size_t update_row_stride, size_t output_row_stride, float epsilon) {
    constexpr int threads = 256;
    add_rms_norm_cuda<<<rows, threads>>>(
        residual, update, weight, output, width, rows, residual_row_stride,
        update_row_stride, output_row_stride, epsilon);
}

void launch_contiguous(
    const float* input, float* output, size_t count, int64_t d0, int64_t d1,
    int64_t d2, size_t s0, size_t s1, size_t s2, size_t s3) {
    constexpr int threads = 256;
    contiguous_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                      threads>>>(
        input, output, count, d0, d1, d2, s0, s1, s2, s3);
}

}  // namespace mollm_cuda
