#include "kernels/attention_x86.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <vector>

namespace {

inline float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline float dot_avx2(const float* lhs, const float* rhs, int count) {
    __m256 first = _mm256_setzero_ps();
    __m256 second = _mm256_setzero_ps();
    int i = 0;
    for (; i + 15 < count; i += 16) {
        first = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i),
                                _mm256_loadu_ps(rhs + i), first);
        second = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i + 8),
                                 _mm256_loadu_ps(rhs + i + 8), second);
    }
    for (; i + 7 < count; i += 8) {
        first = _mm256_fmadd_ps(_mm256_loadu_ps(lhs + i),
                                _mm256_loadu_ps(rhs + i), first);
    }
    float result = horizontal_sum(_mm256_add_ps(first, second));
    for (; i < count; ++i)
        result += lhs[i] * rhs[i];
    return result;
}

inline void softmax(float* values, int count) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < count; ++i)
        maximum = std::max(maximum, values[i]);
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        values[i] = std::exp(values[i] - maximum);
        sum += values[i];
    }
    const float inverse = 1.0f / sum;
    for (int i = 0; i < count; ++i)
        values[i] *= inverse;
}

}  // namespace

void sdpa_x86_avx2_head(const float* query, const float* key,
                        const float* value, float* output, int query_tokens,
                        int key_tokens, int key_dim, int value_dim,
                        float scale, const float* mask) {
    std::vector<float> scores(key_tokens);
    for (int token = 0; token < query_tokens; ++token) {
        const float* query_row =
            query + static_cast<size_t>(token) * key_dim;
        const float* mask_row =
            mask ? mask + static_cast<size_t>(token) * key_tokens : nullptr;
        int visible_keys = key_tokens;
        if (mask_row) {
            while (visible_keys > 0 &&
                   mask_row[visible_keys - 1] ==
                       -std::numeric_limits<float>::infinity()) {
                --visible_keys;
            }
        }
        float* output_row =
            output + static_cast<size_t>(token) * value_dim;
        if (visible_keys == 0) {
            std::memset(output_row, 0,
                        static_cast<size_t>(value_dim) * sizeof(float));
            continue;
        }
        for (int key_token = 0; key_token < visible_keys; ++key_token) {
            scores[key_token] =
                dot_avx2(
                    query_row,
                    key + static_cast<size_t>(key_token) * key_dim,
                    key_dim) *
                scale;
            if (mask_row)
                scores[key_token] += mask_row[key_token];
        }
        softmax(scores.data(), visible_keys);

        std::memset(output_row, 0,
                    static_cast<size_t>(value_dim) * sizeof(float));
        for (int key_token = 0; key_token < visible_keys; ++key_token) {
            const float probability = scores[key_token];
            if (probability == 0.0f)
                continue;
            const __m256 multiplier = _mm256_set1_ps(probability);
            const float* value_row =
                value + static_cast<size_t>(key_token) * value_dim;
            int d = 0;
            for (; d + 7 < value_dim; d += 8) {
                _mm256_storeu_ps(
                    output_row + d,
                    _mm256_fmadd_ps(_mm256_loadu_ps(value_row + d),
                                    multiplier,
                                    _mm256_loadu_ps(output_row + d)));
            }
            for (; d < value_dim; ++d)
                output_row[d] += probability * value_row[d];
        }
    }
}
