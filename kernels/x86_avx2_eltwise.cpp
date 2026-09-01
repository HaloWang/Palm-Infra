#include "kernels/x86_avx2_eltwise.h"

#include <cmath>
#include <immintrin.h>

namespace {

inline float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

// Same 2^f polynomial as the NEON path (~7 bits in [-88, 88]).
inline __m256 sigmoid_avx2(__m256 x) {
    const __m256 neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    const __m256 clamped =
        _mm256_min_ps(_mm256_max_ps(neg, _mm256_set1_ps(-88.0f)),
                      _mm256_set1_ps(88.0f));
    const __m256 t =
        _mm256_mul_ps(clamped, _mm256_set1_ps(1.4426950408889634f));
    const __m256 n = _mm256_floor_ps(t);
    const __m256 f = _mm256_sub_ps(t, n);
    const __m256i ni = _mm256_cvtps_epi32(n);
    const __m256 pow2n = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23));
    __m256 pow2f = _mm256_fmadd_ps(_mm256_set1_ps(0.0096813f), f,
                                   _mm256_set1_ps(0.0555049f));
    pow2f = _mm256_fmadd_ps(pow2f, f, _mm256_set1_ps(0.2402265f));
    pow2f = _mm256_fmadd_ps(pow2f, f, _mm256_set1_ps(0.6931472f));
    pow2f = _mm256_fmadd_ps(pow2f, f, _mm256_set1_ps(1.0f));
    const __m256 ex = _mm256_mul_ps(pow2n, pow2f);
    const __m256 one = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(one, _mm256_add_ps(one, ex));
}

}  // namespace

void rms_norm_x86_avx2(const float* x, const float* weight, float* out, int D,
                       int N, float eps, int ldx, int ldo) {
    for (int n = 0; n < N; ++n) {
        const float* x_row = x + static_cast<size_t>(n) * ldx;
        float* o_row = out + static_cast<size_t>(n) * ldo;
        __m256 sum = _mm256_setzero_ps();
        int d = 0;
        for (; d + 7 < D; d += 8) {
            const __m256 v = _mm256_loadu_ps(x_row + d);
            sum = _mm256_fmadd_ps(v, v, sum);
        }
        float sum_sq = horizontal_sum(sum);
        for (; d < D; ++d)
            sum_sq += x_row[d] * x_row[d];
        const float rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + eps);
        const __m256 rms_v = _mm256_set1_ps(rms);
        d = 0;
        for (; d + 7 < D; d += 8) {
            _mm256_storeu_ps(
                o_row + d,
                _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x_row + d), rms_v),
                              _mm256_loadu_ps(weight + d)));
        }
        for (; d < D; ++d)
            o_row[d] = x_row[d] * rms * weight[d];
    }
}

void add_rms_norm_row_x86_avx2(float* residual, const float* update, float* out,
                               const float* weight, int D, float eps) {
    __m256 sum = _mm256_setzero_ps();
    int d = 0;
    for (; d + 7 < D; d += 8) {
        const __m256 value =
            _mm256_add_ps(_mm256_loadu_ps(residual + d),
                          _mm256_loadu_ps(update + d));
        _mm256_storeu_ps(residual + d, value);
        sum = _mm256_fmadd_ps(value, value, sum);
    }
    float sum_sq = horizontal_sum(sum);
    for (; d < D; ++d) {
        const float value = residual[d] + update[d];
        residual[d] = value;
        sum_sq += value * value;
    }
    const float scale = 1.0f / std::sqrt(sum_sq / static_cast<float>(D) + eps);
    const __m256 scale_v = _mm256_set1_ps(scale);
    d = 0;
    for (; d + 7 < D; d += 8) {
        _mm256_storeu_ps(
            out + d,
            _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(residual + d), scale_v),
                          _mm256_loadu_ps(weight + d)));
    }
    for (; d < D; ++d)
        out[d] = residual[d] * scale * weight[d];
}

void swiglu_row_x86_avx2(const float* gate, const float* up, float* out, int I) {
    int i = 0;
    for (; i + 7 < I; i += 8) {
        const __m256 g = _mm256_loadu_ps(gate + i);
        _mm256_storeu_ps(
            out + i,
            _mm256_mul_ps(_mm256_mul_ps(g, sigmoid_avx2(g)),
                          _mm256_loadu_ps(up + i)));
    }
    for (; i < I; ++i) {
        const float g = gate[i];
        out[i] = (g / (1.0f + std::exp(-g))) * up[i];
    }
}

void sigmoid_mul_row_x86_avx2(const float* value, const float* gate, float* out,
                              int D) {
    int i = 0;
    for (; i + 7 < D; i += 8) {
        _mm256_storeu_ps(
            out + i,
            _mm256_mul_ps(_mm256_loadu_ps(value + i),
                          sigmoid_avx2(_mm256_loadu_ps(gate + i))));
    }
    for (; i < D; ++i)
        out[i] = value[i] / (1.0f + std::exp(-gate[i]));
}
