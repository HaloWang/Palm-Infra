#include "engine/cuda_internal.h"
#include "engine/cuda_reduction.cuh"

#include <cmath>

namespace {

constexpr int shortconv_max_kernel = 8;
constexpr int gdn_head_dim = 128;

__global__ void shortconv_cuda(
    const float* input, const float* weight, float* state, float* output,
    int groups, int sequence_length, int kernel_size, int real_tokens,
    size_t input_feature_stride, size_t input_row_stride) {
    const int group = static_cast<int>(blockIdx.x) * blockDim.x +
        static_cast<int>(threadIdx.x);
    if (group >= groups)
        return;

    const int prefix_length = kernel_size - 1;
    float window[shortconv_max_kernel - 1];
    float* group_state = state +
        static_cast<size_t>(group) * prefix_length;
#pragma unroll
    for (int index = 0; index < shortconv_max_kernel - 1; ++index)
        if (index < prefix_length)
            window[index] = group_state[index];

    const float* group_weight = weight +
        static_cast<size_t>(group) * kernel_size;
    float* group_output = output +
        static_cast<size_t>(group) * sequence_length;
    for (int token = 0; token < real_tokens; ++token) {
        const float value = input[
            static_cast<size_t>(token) * input_row_stride +
            static_cast<size_t>(group) * input_feature_stride];
        float sum = value * group_weight[prefix_length];
#pragma unroll
        for (int index = 0; index < shortconv_max_kernel - 1; ++index)
            if (index < prefix_length)
                sum = fmaf(window[index], group_weight[index], sum);
        group_output[token] = sum / (1.0f + expf(-sum));
#pragma unroll
        for (int index = 0; index < shortconv_max_kernel - 2; ++index)
            if (index + 1 < prefix_length)
                window[index] = window[index + 1];
        if (prefix_length > 0)
            window[prefix_length - 1] = value;
    }
    for (int token = real_tokens; token < sequence_length; ++token)
        group_output[token] = 0.0f;
#pragma unroll
    for (int index = 0; index < shortconv_max_kernel - 1; ++index)
        if (index < prefix_length)
            group_state[index] = window[index];
}

__device__ __forceinline__ float gdn_softplus(float value) {
    if (value > 20.0f)
        return value;
    if (value < -20.0f)
        return expf(value);
    return log1pf(expf(value));
}

// Qwen3.5's recurrent dimensions are 128-wide. One block owns one value head
// so every state column is independent across threads while the token axis is
// traversed in recurrence order. Q/K and RMS reductions stay on chip.
__global__ void gdn_128_cuda(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int num_value_heads,
    int sequence_length, int real_tokens, bool normalize_qk, float rms_epsilon,
    float l2_epsilon, float scale, size_t a_row_stride,
    size_t b_row_stride, size_t z_row_stride) {
    const int value_head = static_cast<int>(blockIdx.x);
    const int dimension = static_cast<int>(threadIdx.x);
    if (value_head >= num_value_heads)
        return;
    const int repeat = num_value_heads / num_heads;
    const int key_head = value_head / repeat;
    const int qkv_key_width = num_heads * gdn_head_dim;
    const int output_width = num_value_heads * gdn_head_dim;
    float* head_state = state +
        static_cast<size_t>(value_head) * gdn_head_dim * gdn_head_dim;

    __shared__ float query[gdn_head_dim];
    __shared__ float key[gdn_head_dim];
    __shared__ float reduction[gdn_head_dim];
    __shared__ float gate[2];

    for (int token = 0; token < sequence_length; ++token) {
        if (token >= real_tokens) {
            output[static_cast<size_t>(token) * output_width +
                   static_cast<size_t>(value_head) * gdn_head_dim +
                   dimension] = 0.0f;
            continue;
        }

        const size_t query_index =
            (static_cast<size_t>(key_head) * gdn_head_dim + dimension) *
                sequence_length + token;
        const size_t key_index =
            (static_cast<size_t>(qkv_key_width) +
             static_cast<size_t>(key_head) * gdn_head_dim + dimension) *
                sequence_length + token;
        query[dimension] = qkv[query_index];
        key[dimension] = qkv[key_index];
        float query_inverse = 1.0f;
        float key_inverse = 1.0f;
        if (normalize_qk) {
            const float query_sum =
                mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                    query[dimension] * query[dimension], reduction);
            const float key_sum =
                mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                    key[dimension] * key[dimension], reduction);
            query_inverse = rsqrtf(query_sum + l2_epsilon);
            key_inverse = rsqrtf(key_sum + l2_epsilon);
        }
        query[dimension] *= query_inverse;
        key[dimension] *= key_inverse;
        __syncthreads();
        const float query_key =
            mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                query[dimension] * key[dimension], reduction);

        if (dimension == 0) {
            const float a_value =
                a[static_cast<size_t>(token) * a_row_stride + value_head];
            const float b_value =
                b[static_cast<size_t>(token) * b_row_stride + value_head];
            gate[0] = expf(
                -expf(a_log[value_head]) *
                gdn_softplus(a_value + dt_bias[value_head]));
            gate[1] = 1.0f / (1.0f + expf(-b_value));
        }
        __syncthreads();

        float key_value = 0.0f;
        float attention_decay = 0.0f;
        for (int state_row = 0; state_row < gdn_head_dim; ++state_row) {
            const float decayed =
                head_state[state_row * gdn_head_dim + dimension] * gate[0];
            key_value = fmaf(decayed, key[state_row], key_value);
            attention_decay =
                fmaf(decayed, query[state_row], attention_decay);
        }
        const size_t value_index =
            (static_cast<size_t>(2 * qkv_key_width) +
             static_cast<size_t>(value_head) * gdn_head_dim + dimension) *
                sequence_length + token;
        const float delta = (qkv[value_index] - key_value) * gate[1];
        for (int state_row = 0; state_row < gdn_head_dim; ++state_row) {
            const size_t index =
                static_cast<size_t>(state_row) * gdn_head_dim + dimension;
            head_state[index] =
                head_state[index] * gate[0] + key[state_row] * delta;
        }
        const float attention =
            (attention_decay + delta * query_key) * scale;
        const float square_sum =
            mollm_cuda::detail::block_reduce_sum<gdn_head_dim>(
                attention * attention, reduction);
        const float inverse_rms =
            rsqrtf(square_sum / gdn_head_dim + rms_epsilon);
        const float z_value = z[
            static_cast<size_t>(token) * z_row_stride +
            static_cast<size_t>(value_head) * gdn_head_dim + dimension];
        output[static_cast<size_t>(token) * output_width +
               static_cast<size_t>(value_head) * gdn_head_dim + dimension] =
            attention * inverse_rms * norm_weight[dimension] *
            (z_value / (1.0f + expf(-z_value)));
        __syncthreads();
    }
}

}  // namespace

namespace mollm_cuda {

bool launch_shortconv(
    const float* input, const float* weight, float* state, float* output,
    int groups, int sequence_length, int kernel_size, int real_tokens,
    size_t input_feature_stride, size_t input_row_stride) {
    if (!input || !weight || !state || !output || groups <= 0 ||
        sequence_length <= 0 || kernel_size <= 0 ||
        kernel_size > shortconv_max_kernel)
        return false;
    const int process_length =
        real_tokens > 0 && real_tokens < sequence_length
        ? real_tokens : sequence_length;
    constexpr int threads = 256;
    shortconv_cuda<<<(groups + threads - 1) / threads, threads>>>(
        input, weight, state, output, groups, sequence_length, kernel_size,
        process_length, input_feature_stride, input_row_stride);
    return true;
}

bool launch_gdn(
    const float* qkv, const float* a, const float* b, const float* z,
    const float* a_log, const float* dt_bias, const float* norm_weight,
    float* state, float* output, int num_heads, int num_value_heads,
    int key_dimension, int value_dimension, int sequence_length,
    int real_tokens, bool normalize_qk, float rms_epsilon, float l2_epsilon,
    float scale, size_t a_row_stride, size_t b_row_stride,
    size_t z_row_stride) {
    if (!qkv || !a || !b || !z || !a_log || !dt_bias || !norm_weight ||
        !state || !output || num_heads <= 0 || num_value_heads <= 0 ||
        num_value_heads % num_heads != 0 ||
        key_dimension != gdn_head_dim || value_dimension != gdn_head_dim ||
        sequence_length <= 0)
        return false;
    const int process_length =
        real_tokens > 0 && real_tokens < sequence_length
        ? real_tokens : sequence_length;
    gdn_128_cuda<<<num_value_heads, gdn_head_dim>>>(
        qkv, a, b, z, a_log, dt_bias, norm_weight, state, output, num_heads,
        num_value_heads, sequence_length, process_length, normalize_qk,
        rms_epsilon, l2_epsilon, scale, a_row_stride, b_row_stride,
        z_row_stride);
    return true;
}

}  // namespace mollm_cuda
