#include "engine/cuda_internal.h"
#include "engine/cuda_reduction.cuh"

#include <cfloat>
#include <cmath>

namespace {

__global__ void rope_cuda(
    const float* input, const float* cosine, const float* sine,
    float* output, int feature_dim, int sequence_length, int channels,
    int shape2, int rope_dim, bool interleave,
    size_t x_s0, size_t x_s1, size_t x_s2, size_t x_s3,
    size_t c_s0, size_t c_s1, size_t s_s0, size_t s_s1,
    size_t o_s0, size_t o_s1, size_t o_s2, size_t o_s3) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(feature_dim) *
        sequence_length * channels;
    if (index >= count)
        return;
    size_t remaining = index;
    const int dimension = static_cast<int>(remaining % feature_dim);
    remaining /= feature_dim;
    const int position = static_cast<int>(remaining % sequence_length);
    const int channel = static_cast<int>(remaining / sequence_length);
    const int channel2 = channel % shape2;
    const int channel3 = channel / shape2;
    const size_t input_base = static_cast<size_t>(position) * x_s1 +
        static_cast<size_t>(channel2) * x_s2 +
        static_cast<size_t>(channel3) * x_s3;
    const size_t output_index = static_cast<size_t>(dimension) * o_s0 +
        static_cast<size_t>(position) * o_s1 +
        static_cast<size_t>(channel2) * o_s2 +
        static_cast<size_t>(channel3) * o_s3;
    if (dimension >= rope_dim) {
        output[output_index] =
            input[input_base + static_cast<size_t>(dimension) * x_s0];
        return;
    }

    const int half = rope_dim / 2;
    const int pair = interleave ? dimension / 2 : dimension % half;
    const int first = interleave ? pair * 2 : pair;
    const int second = interleave ? first + 1 : pair + half;
    const float x0 = input[input_base + static_cast<size_t>(first) * x_s0];
    const float x1 = input[input_base + static_cast<size_t>(second) * x_s0];
    const float c = cosine[static_cast<size_t>(position) * c_s1 +
                           static_cast<size_t>(pair) * c_s0];
    const float s = sine[static_cast<size_t>(position) * s_s1 +
                         static_cast<size_t>(pair) * s_s0];
    const bool first_component = interleave
        ? (dimension & 1) == 0 : dimension < half;
    output[output_index] = first_component
        ? x0 * c - x1 * s : x0 * s + x1 * c;
}

__global__ void qk_rms_norm_rope_cuda(
    const float* query, const float* key, const float* query_weight,
    const float* key_weight, const float* cosine, const float* sine,
    float* output, int feature_dim, int sequence_length, int query_heads,
    int total_heads, int rope_dim, bool interleave,
    size_t query_feature_stride, size_t query_row_stride,
    size_t key_feature_stride, size_t key_row_stride,
    size_t cosine_feature_stride, size_t cosine_position_stride,
    size_t sine_feature_stride, size_t sine_position_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride, float epsilon) {
    const int row = static_cast<int>(blockIdx.x);
    const int rows = sequence_length * total_heads;
    if (row >= rows)
        return;

    const int query_rows = sequence_length * query_heads;
    const bool is_query = row < query_rows;
    const int source_row = is_query ? row : row - query_rows;
    const float* source = (is_query ? query : key) +
        static_cast<size_t>(source_row) *
            (is_query ? query_row_stride : key_row_stride);
    const size_t source_feature_stride = is_query
        ? query_feature_stride : key_feature_stride;
    const float* weight = is_query ? query_weight : key_weight;

    float sum = 0.0f;
    for (int dimension = threadIdx.x; dimension < feature_dim;
         dimension += blockDim.x) {
        const float value = source[
            static_cast<size_t>(dimension) * source_feature_stride];
        sum += value * value;
    }
    __shared__ float reduction[256];
    const float block_sum =
        mollm_cuda::detail::block_reduce_sum_256(sum, reduction);
    const float inverse = rsqrtf(block_sum / feature_dim + epsilon);

    const int position = source_row % sequence_length;
    const int head = row / sequence_length;
    float* destination = output +
        static_cast<size_t>(position) * output_position_stride +
        static_cast<size_t>(head) * output_head_stride;
    const int half = rope_dim / 2;
    for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
        const int first = interleave ? pair * 2 : pair;
        const int second = interleave ? first + 1 : pair + half;
        const float x0 = source[
            static_cast<size_t>(first) * source_feature_stride] * inverse *
            weight[first];
        const float x1 = source[
            static_cast<size_t>(second) * source_feature_stride] * inverse *
            weight[second];
        const float c = cosine[
            static_cast<size_t>(pair) * cosine_feature_stride +
            static_cast<size_t>(position) * cosine_position_stride];
        const float s = sine[
            static_cast<size_t>(pair) * sine_feature_stride +
            static_cast<size_t>(position) * sine_position_stride];
        destination[static_cast<size_t>(first) * output_feature_stride] =
            x0 * c - x1 * s;
        destination[static_cast<size_t>(second) * output_feature_stride] =
            x1 * c + x0 * s;
    }
    for (int dimension = rope_dim + threadIdx.x; dimension < feature_dim;
         dimension += blockDim.x) {
        destination[static_cast<size_t>(dimension) * output_feature_stride] =
            source[static_cast<size_t>(dimension) * source_feature_stride] *
            inverse * weight[dimension];
    }
}

__device__ float load_kv_cache_value(
    const void* cache, size_t index, bool fp16_cache) {
    return fp16_cache
        ? __half2float(static_cast<const __half*>(cache)[index])
        : static_cast<const float*>(cache)[index];
}

__device__ void store_kv_cache_value(
    void* cache, size_t index, float value, bool fp16_cache) {
    if (fp16_cache)
        static_cast<__half*>(cache)[index] = __float2half_rn(value);
    else
        static_cast<float*>(cache)[index] = value;
}

__global__ void append_kv_cuda(
    const float* key, const float* value, void* key_cache,
    void* value_cache, bool fp16_cache, int num_kv_heads,
    int current_length, int past_length, int max_length,
    int key_dim, int value_dim,
    size_t key_position_stride, size_t key_head_stride,
    size_t value_position_stride, size_t value_head_stride) {
    const int maximum_dim = max(key_dim, value_dim);
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(num_kv_heads) *
        current_length * maximum_dim;
    if (index >= count)
        return;
    size_t remaining = index;
    const int dimension = static_cast<int>(remaining % maximum_dim);
    remaining /= maximum_dim;
    const int position = static_cast<int>(remaining % current_length);
    const int head = static_cast<int>(remaining / current_length);
    if (dimension < key_dim) {
        const size_t destination =
            (static_cast<size_t>(head) * max_length + past_length +
             position) * key_dim + dimension;
        store_kv_cache_value(
            key_cache, destination,
            key[static_cast<size_t>(head) * key_head_stride +
                static_cast<size_t>(position) * key_position_stride +
                dimension],
            fp16_cache);
    }
    if (dimension < value_dim) {
        const size_t destination =
            (static_cast<size_t>(head) * max_length + past_length +
             position) * value_dim + dimension;
        store_kv_cache_value(
            value_cache, destination,
            value[static_cast<size_t>(head) * value_head_stride +
                  static_cast<size_t>(position) * value_position_stride +
                  dimension],
            fp16_cache);
    }
}

__global__ void sdpa_scores_cuda(
    const float* query, const void* key, float* scores,
    const float* mask, int num_heads, int num_kv_heads,
    int query_length, int key_length, int past_length, int key_dim,
    int key_capacity, bool cached, bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_position_stride,
    size_t query_head_stride, size_t key_feature_stride,
    size_t key_position_stride, size_t key_head_stride,
    size_t mask_column_stride, size_t mask_row_stride) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;
    const size_t count = static_cast<size_t>(num_heads) * query_length *
        key_length;
    if (index >= count)
        return;
    size_t remaining = index;
    const int key_position = static_cast<int>(remaining % key_length);
    remaining /= key_length;
    const int query_position = static_cast<int>(remaining % query_length);
    const int head = static_cast<int>(remaining / query_length);
    const int key_head = head / (num_heads / num_kv_heads);
    const float* query_row = query +
        static_cast<size_t>(head) * query_head_stride +
        static_cast<size_t>(query_position) * query_position_stride;
    const size_t cached_key_base =
        (static_cast<size_t>(key_head) * key_capacity + key_position) *
        key_dim;
    const auto* current_key = static_cast<const float*>(key);
    const size_t current_key_base =
        static_cast<size_t>(key_head) * key_head_stride +
        static_cast<size_t>(key_position) * key_position_stride;
    float dot = 0.0f;
    for (int dimension = 0; dimension < key_dim; ++dimension) {
        const float key_value = cached
            ? load_kv_cache_value(
                  key, cached_key_base + dimension, fp16_cache)
            : current_key[current_key_base +
                          static_cast<size_t>(dimension) *
                              key_feature_stride];
        dot += query_row[static_cast<size_t>(dimension) *
                         query_feature_stride] * key_value;
    }
    float score = dot * scale;
    if (mask) {
        score += mask[static_cast<size_t>(query_position) * mask_row_stride +
                      static_cast<size_t>(key_position) *
                          mask_column_stride];
    } else if (causal && key_position > past_length + query_position) {
        score = -FLT_MAX;
    }
    scores[index] = score;
}

__global__ void sdpa_output_cuda(
    const float* scores, const void* value, float* output,
    int num_heads, int num_kv_heads, int query_length, int key_length,
    int value_dim, int value_capacity, bool cached, bool fp16_cache,
    size_t value_feature_stride, size_t value_position_stride,
    size_t value_head_stride, size_t output_feature_stride,
    size_t output_position_stride, size_t output_head_stride) {
    const int row = blockIdx.x;
    const int head = row / query_length;
    const int query_position = row % query_length;
    if (head >= num_heads)
        return;
    const int key_head = head / (num_heads / num_kv_heads);
    const float* score_row = scores + static_cast<size_t>(row) * key_length;
    __shared__ float reduction[256];
    float local_maximum = -FLT_MAX;
    for (int position = threadIdx.x; position < key_length;
         position += blockDim.x)
        local_maximum = fmaxf(local_maximum, score_row[position]);
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];
    float local_sum = 0.0f;
    for (int position = threadIdx.x; position < key_length;
         position += blockDim.x)
        local_sum += expf(score_row[position] - maximum);
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_sum = reduction[0] > 0.0f
        ? 1.0f / reduction[0] : 0.0f;
    for (int dimension = threadIdx.x; dimension < value_dim;
         dimension += blockDim.x) {
        float result = 0.0f;
        for (int position = 0; position < key_length; ++position) {
            const float probability =
                expf(score_row[position] - maximum) * inverse_sum;
            const size_t cached_index =
                (static_cast<size_t>(key_head) * value_capacity + position) *
                    value_dim + dimension;
            const auto* current_value = static_cast<const float*>(value);
            const size_t current_index =
                static_cast<size_t>(key_head) * value_head_stride +
                static_cast<size_t>(position) * value_position_stride +
                static_cast<size_t>(dimension) * value_feature_stride;
            const float value_element = cached
                ? load_kv_cache_value(value, cached_index, fp16_cache)
                : current_value[current_index];
            result += probability * value_element;
        }
        output[static_cast<size_t>(head) * output_head_stride +
               static_cast<size_t>(query_position) *
                   output_position_stride +
               static_cast<size_t>(dimension) * output_feature_stride] =
            result;
    }
}

// Decode has one query position, so the score and softmax/value passes can be
// owned by one block per query head.  Keeping the temporary score row in the
// backend scratch buffer avoids a context-length shared-memory limit while a
// single launch removes the inter-kernel boundary.  Softmax exponentials are
// also computed once per key instead of once per output dimension.
__global__ void sdpa_decode_cuda(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int key_length, int past_length, int key_dim, int value_dim,
    int key_capacity, bool cached, bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_head_stride,
    size_t key_feature_stride, size_t key_position_stride,
    size_t key_head_stride, size_t value_feature_stride,
    size_t value_position_stride, size_t value_head_stride,
    size_t mask_column_stride, size_t output_feature_stride,
    size_t output_head_stride) {
    const int head = blockIdx.x;
    if (head >= num_heads)
        return;
    const int key_head = head / (num_heads / num_kv_heads);
    const float* query_row = query +
        static_cast<size_t>(head) * query_head_stride;
    float* score_row = scores + static_cast<size_t>(head) * key_length;
    __shared__ float reduction[256];

    float local_maximum = -FLT_MAX;
    for (int key_position = threadIdx.x; key_position < key_length;
         key_position += blockDim.x) {
        const size_t cached_key_base =
            (static_cast<size_t>(key_head) * key_capacity + key_position) *
            key_dim;
        const size_t current_key_base =
            static_cast<size_t>(key_head) * key_head_stride +
            static_cast<size_t>(key_position) * key_position_stride;
        float dot = 0.0f;
        for (int dimension = 0; dimension < key_dim; ++dimension) {
            const float key_element = cached
                ? load_kv_cache_value(
                      key, cached_key_base + dimension, fp16_cache)
                : static_cast<const float*>(key)[
                      current_key_base + static_cast<size_t>(dimension) *
                          key_feature_stride];
            dot += query_row[static_cast<size_t>(dimension) *
                             query_feature_stride] * key_element;
        }
        float score = dot * scale;
        if (mask)
            score += mask[static_cast<size_t>(key_position) *
                          mask_column_stride];
        else if (causal && key_position > past_length)
            score = -FLT_MAX;
        score_row[key_position] = score;
        local_maximum = fmaxf(local_maximum, score);
    }
    reduction[threadIdx.x] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        __syncthreads();
    }
    const float maximum = reduction[0];

    float local_sum = 0.0f;
    for (int key_position = threadIdx.x; key_position < key_length;
         key_position += blockDim.x) {
        const float numerator = expf(score_row[key_position] - maximum);
        score_row[key_position] = numerator;
        local_sum += numerator;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse_sum = reduction[0] > 0.0f
        ? 1.0f / reduction[0] : 0.0f;

    for (int dimension = threadIdx.x; dimension < value_dim;
         dimension += blockDim.x) {
        float result = 0.0f;
        for (int key_position = 0; key_position < key_length;
             ++key_position) {
            const size_t cached_value_index =
                (static_cast<size_t>(key_head) * key_capacity +
                 key_position) * value_dim + dimension;
            const size_t current_value_index =
                static_cast<size_t>(key_head) * value_head_stride +
                static_cast<size_t>(key_position) * value_position_stride +
                static_cast<size_t>(dimension) * value_feature_stride;
            const float value_element = cached
                ? load_kv_cache_value(
                      value, cached_value_index, fp16_cache)
                : static_cast<const float*>(value)[current_value_index];
            result += score_row[key_position] * inverse_sum * value_element;
        }
        output[static_cast<size_t>(head) * output_head_stride +
               static_cast<size_t>(dimension) * output_feature_stride] =
            result;
    }
}

}  // namespace

namespace mollm_cuda {

void launch_rope(
    const float* input, const float* cosine, const float* sine, float* output,
    int feature_dim, int sequence_length, int channels, int shape2,
    int rope_dim, bool interleave, size_t x_s0, size_t x_s1, size_t x_s2,
    size_t x_s3, size_t c_s0, size_t c_s1, size_t s_s0, size_t s_s1,
    size_t o_s0, size_t o_s1, size_t o_s2, size_t o_s3) {
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(feature_dim) *
        sequence_length * channels;
    rope_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                threads>>>(
        input, cosine, sine, output, feature_dim, sequence_length, channels,
        shape2, rope_dim, interleave, x_s0, x_s1, x_s2, x_s3, c_s0, c_s1,
        s_s0, s_s1, o_s0, o_s1, o_s2, o_s3);
}

void launch_qk_rms_norm_rope(
    const float* query, const float* key, const float* query_weight,
    const float* key_weight, const float* cosine, const float* sine,
    float* output, int feature_dim, int sequence_length, int query_heads,
    int total_heads, int rope_dim, bool interleave,
    size_t query_feature_stride, size_t query_row_stride,
    size_t key_feature_stride, size_t key_row_stride,
    size_t cosine_feature_stride, size_t cosine_position_stride,
    size_t sine_feature_stride, size_t sine_position_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride, float epsilon) {
    constexpr int threads = 256;
    qk_rms_norm_rope_cuda<<<sequence_length * total_heads, threads>>>(
        query, key, query_weight, key_weight, cosine, sine, output,
        feature_dim, sequence_length, query_heads, total_heads, rope_dim,
        interleave, query_feature_stride, query_row_stride,
        key_feature_stride, key_row_stride, cosine_feature_stride,
        cosine_position_stride, sine_feature_stride, sine_position_stride,
        output_feature_stride, output_position_stride, output_head_stride,
        epsilon);
}

void launch_append_kv(
    const float* key, const float* value, void* key_cache,
    void* value_cache, bool fp16_cache, int num_kv_heads,
    int current_length, int past_length, int max_length,
    int key_dim, int value_dim, size_t key_position_stride,
    size_t key_head_stride, size_t value_position_stride,
    size_t value_head_stride) {
    constexpr int threads = 256;
    const int maximum_dim = key_dim > value_dim ? key_dim : value_dim;
    const size_t count = static_cast<size_t>(num_kv_heads) * current_length *
        maximum_dim;
    append_kv_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                     threads>>>(
        key, value, key_cache, value_cache, fp16_cache, num_kv_heads,
        current_length, past_length, max_length, key_dim, value_dim,
        key_position_stride, key_head_stride, value_position_stride,
        value_head_stride);
}

void launch_sdpa_scores(
    const float* query, const void* key, float* scores, const float* mask,
    int num_heads, int num_kv_heads, int query_length, int key_length,
    int past_length, int key_dim, int key_capacity, bool cached,
    bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_position_stride,
    size_t query_head_stride, size_t key_feature_stride,
    size_t key_position_stride, size_t key_head_stride,
    size_t mask_column_stride, size_t mask_row_stride) {
    constexpr int threads = 256;
    const size_t count = static_cast<size_t>(num_heads) * query_length *
        key_length;
    sdpa_scores_cuda<<<static_cast<unsigned>((count + threads - 1) / threads),
                       threads>>>(
        query, key, scores, mask, num_heads, num_kv_heads, query_length,
        key_length, past_length, key_dim, key_capacity, cached, fp16_cache,
        causal, scale, query_feature_stride, query_position_stride,
        query_head_stride, key_feature_stride, key_position_stride,
        key_head_stride, mask_column_stride, mask_row_stride);
}

void launch_sdpa_output(
    const float* scores, const void* value, float* output, int num_heads,
    int num_kv_heads, int query_length, int key_length, int value_dim,
    int value_capacity, bool cached, bool fp16_cache,
    size_t value_feature_stride, size_t value_position_stride,
    size_t value_head_stride,
    size_t output_feature_stride, size_t output_position_stride,
    size_t output_head_stride) {
    constexpr int threads = 256;
    sdpa_output_cuda<<<num_heads * query_length, threads>>>(
        scores, value, output, num_heads, num_kv_heads, query_length,
        key_length, value_dim, value_capacity, cached, fp16_cache,
        value_feature_stride, value_position_stride, value_head_stride,
        output_feature_stride, output_position_stride, output_head_stride);
}

void launch_sdpa_decode(
    const float* query, const void* key, const void* value, float* scores,
    float* output, const float* mask, int num_heads, int num_kv_heads,
    int key_length, int past_length, int key_dim, int value_dim,
    int key_capacity, bool cached, bool fp16_cache, bool causal, float scale,
    size_t query_feature_stride, size_t query_head_stride,
    size_t key_feature_stride, size_t key_position_stride,
    size_t key_head_stride, size_t value_feature_stride,
    size_t value_position_stride, size_t value_head_stride,
    size_t mask_column_stride, size_t output_feature_stride,
    size_t output_head_stride) {
    constexpr int threads = 256;
    sdpa_decode_cuda<<<num_heads, threads>>>(
        query, key, value, scores, output, mask, num_heads, num_kv_heads,
        key_length, past_length, key_dim, value_dim, key_capacity, cached,
        fp16_cache, causal, scale, query_feature_stride, query_head_stride,
        key_feature_stride, key_position_stride, key_head_stride,
        value_feature_stride, value_position_stride, value_head_stride,
        mask_column_stride, output_feature_stride, output_head_stride);
}

}  // namespace mollm_cuda
