#include "kernels/x86_avx2.h"
#include "kernels/matmul_internal.h"

#include <algorithm>
#include <immintrin.h>

namespace mollm::cpu::x86 {
namespace {

inline float horizontal_sum(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

struct Q4Vectors {
    __m256 values[4];
};

inline Q4Vectors unpack_q4_block32(const uint8_t* packed) {
    const __m128i bytes =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i sign_bit = _mm_set1_epi8(0x08);
    const __m128i low = _mm_and_si128(bytes, nibble_mask);
    const __m128i high = _mm_and_si128(
        _mm_srli_epi16(bytes, 4), nibble_mask);
    __m128i q0 = _mm_unpacklo_epi8(low, high);
    __m128i q1 = _mm_unpackhi_epi8(low, high);
    q0 = _mm_sub_epi8(_mm_xor_si128(q0, sign_bit), sign_bit);
    q1 = _mm_sub_epi8(_mm_xor_si128(q1, sign_bit), sign_bit);

    Q4Vectors result;
    result.values[0] =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q0));
    result.values[1] = _mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(q0, 8)));
    result.values[2] =
        _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q1));
    result.values[3] = _mm256_cvtepi32_ps(
        _mm256_cvtepi8_epi32(_mm_srli_si128(q1, 8)));
    return result;
}

inline __m256 dot_q4_block32(const float* activation,
                             const Q4Vectors& weights) {
    __m256 acc = _mm256_mul_ps(_mm256_loadu_ps(activation),
                               weights.values[0]);
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(activation + 8),
                           weights.values[1], acc);
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(activation + 16),
                           weights.values[2], acc);
    return _mm256_fmadd_ps(_mm256_loadu_ps(activation + 24),
                            weights.values[3], acc);
}

inline __m256 dot_q4_loaded(const __m256 a0, const __m256 a1, const __m256 a2,
                            const __m256 a3, const Q4Vectors& weights) {
    __m256 acc = _mm256_mul_ps(a0, weights.values[0]);
    acc = _mm256_fmadd_ps(a1, weights.values[1], acc);
    acc = _mm256_fmadd_ps(a2, weights.values[2], acc);
    return _mm256_fmadd_ps(a3, weights.values[3], acc);
}

inline void load_a32(const float* activation, __m256& a0, __m256& a1,
                     __m256& a2, __m256& a3) {
    a0 = _mm256_loadu_ps(activation);
    a1 = _mm256_loadu_ps(activation + 8);
    a2 = _mm256_loadu_ps(activation + 16);
    a3 = _mm256_loadu_ps(activation + 24);
}

constexpr int kInt4NTile = 8;
constexpr int kDenseNTile = 4;
constexpr int kMTile = 4;

inline void gemv_bg32_one_n(const float* a_row, float* out, int n,
                            const Q4B8G32Block* bg32, int groups) {
    float sum = 0.0f;
    const int lane = n & 7;
    const size_t tile = static_cast<size_t>(n / 8);
    for (int group = 0; group < groups; ++group) {
        const auto& block = bg32[tile * groups + group];
        sum += horizontal_sum(dot_q4_block32(
                   a_row + static_cast<size_t>(group) * 32,
                   unpack_q4_block32(block.q[lane]))) *
               block.scales[lane];
    }
    out[n] = sum;
}

inline void gemv_bg128_one_n(const float* a_row, float* out, int n,
                             const Q4B8G128Block* bg128, int groups) {
    float sum = 0.0f;
    const int lane = n & 7;
    const size_t tile = static_cast<size_t>(n / 8);
    for (int group = 0; group < groups; ++group) {
        const auto& block = bg128[tile * groups + group];
        __m256 acc = _mm256_setzero_ps();
        for (int sub = 0; sub < 4; ++sub) {
            acc = _mm256_add_ps(
                acc, dot_q4_block32(a_row + static_cast<size_t>(group) * 128 +
                                        sub * 32,
                                    unpack_q4_block32(block.q[sub][lane])));
        }
        sum += horizontal_sum(acc) * block.scales[lane];
    }
    out[n] = sum;
}

inline void gemv_bg32_tile8(const float* a_row, float* out, int n,
                            const Q4B8G32Block* bg32, int groups) {
    float sums[kInt4NTile] = {};
    const size_t tile = static_cast<size_t>(n / 8);
    for (int group = 0; group < groups; ++group) {
        const auto& block = bg32[tile * groups + group];
        __m256 a0, a1, a2, a3;
        load_a32(a_row + static_cast<size_t>(group) * 32, a0, a1, a2, a3);
        float dots[kInt4NTile];
        for (int lane = 0; lane < kInt4NTile; ++lane) {
            dots[lane] = horizontal_sum(dot_q4_loaded(
                a0, a1, a2, a3, unpack_q4_block32(block.q[lane])));
        }
        _mm256_storeu_ps(
            sums, _mm256_fmadd_ps(_mm256_loadu_ps(dots),
                                  _mm256_loadu_ps(block.scales),
                                  _mm256_loadu_ps(sums)));
    }
    _mm256_storeu_ps(out + n, _mm256_loadu_ps(sums));
}

inline void gemv_bg128_tile8(const float* a_row, float* out, int n,
                             const Q4B8G128Block* bg128, int groups) {
    float sums[kInt4NTile] = {};
    const size_t tile = static_cast<size_t>(n / 8);
    for (int group = 0; group < groups; ++group) {
        const auto& block = bg128[tile * groups + group];
        __m256 acc[kInt4NTile] = {
            _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
            _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int sub = 0; sub < 4; ++sub) {
            __m256 a0, a1, a2, a3;
            load_a32(a_row + static_cast<size_t>(group) * 128 + sub * 32, a0,
                     a1, a2, a3);
            for (int lane = 0; lane < kInt4NTile; ++lane) {
                acc[lane] = _mm256_add_ps(
                    acc[lane],
                    dot_q4_loaded(a0, a1, a2, a3,
                                  unpack_q4_block32(block.q[sub][lane])));
            }
        }
        float dots[kInt4NTile];
        for (int lane = 0; lane < kInt4NTile; ++lane)
            dots[lane] = horizontal_sum(acc[lane]);
        _mm256_storeu_ps(
            sums, _mm256_fmadd_ps(_mm256_loadu_ps(dots),
                                  _mm256_loadu_ps(block.scales),
                                  _mm256_loadu_ps(sums)));
    }
    _mm256_storeu_ps(out + n, _mm256_loadu_ps(sums));
}

inline void gemm_bg32_one_n(const Tensor& A, Tensor& C, int lda, int ldc,
                            int m_base, int m_count, int n,
                            const Q4B8G32Block* bg32, int groups) {
    float sums[kMTile] = {};
    const int lane = n & 7;
    const size_t tile = static_cast<size_t>(n / 8);
    for (int group = 0; group < groups; ++group) {
        const auto& block = bg32[tile * groups + group];
        const Q4Vectors weights = unpack_q4_block32(block.q[lane]);
        const float scale = block.scales[lane];
        for (int mi = 0; mi < m_count; ++mi) {
            const float* activation =
                A.ptr<float>() + static_cast<size_t>(m_base + mi) * lda +
                static_cast<size_t>(group) * 32;
            sums[mi] +=
                horizontal_sum(dot_q4_block32(activation, weights)) * scale;
        }
    }
    for (int mi = 0; mi < m_count; ++mi) {
        C.ptr<float>()[static_cast<size_t>(m_base + mi) * ldc + n] = sums[mi];
    }
}

inline void gemm_bg128_one_n(const Tensor& A, Tensor& C, int lda, int ldc,
                             int m_base, int m_count, int n,
                             const Q4B8G128Block* bg128, int groups) {
    float sums[kMTile] = {};
    const int lane = n & 7;
    const size_t tile = static_cast<size_t>(n / 8);
    for (int group = 0; group < groups; ++group) {
        const auto& block = bg128[tile * groups + group];
        __m256 acc[kMTile] = {_mm256_setzero_ps(), _mm256_setzero_ps(),
                              _mm256_setzero_ps(), _mm256_setzero_ps()};
        for (int sub = 0; sub < 4; ++sub) {
            const Q4Vectors weights =
                unpack_q4_block32(block.q[sub][lane]);
            for (int mi = 0; mi < m_count; ++mi) {
                const float* activation =
                    A.ptr<float>() + static_cast<size_t>(m_base + mi) * lda +
                    static_cast<size_t>(group) * 128 + sub * 32;
                acc[mi] =
                    _mm256_add_ps(acc[mi], dot_q4_block32(activation, weights));
            }
        }
        for (int mi = 0; mi < m_count; ++mi)
            sums[mi] += horizontal_sum(acc[mi]) * block.scales[lane];
    }
    for (int mi = 0; mi < m_count; ++mi) {
        C.ptr<float>()[static_cast<size_t>(m_base + mi) * ldc + n] = sums[mi];
    }
}

template <typename OneN, typename Tile8>
inline void run_n_tiles(int n_begin, int n_end, OneN&& one_n, Tile8&& tile8) {
    int n = n_begin;
    while (n < n_end && (n & 7)) {
        one_n(n);
        ++n;
    }
    for (; n + kInt4NTile <= n_end; n += kInt4NTile)
        tile8(n);
    for (; n < n_end; ++n)
        one_n(n);
}

}  // namespace

void matmul_fp32_avx2_range(const float* A, const float* B, float* C, int N,
                            int K, int lda, int K_weight, int ldc,
                            int m_begin, int m_end) {
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = A + static_cast<size_t>(m) * lda;
        float* c_row = C + static_cast<size_t>(m) * ldc;
        int n = 0;
        for (; n + kDenseNTile <= N; n += kDenseNTile) {
            const float* b_rows[kDenseNTile] = {
                B + static_cast<size_t>(n) * K_weight,
                B + static_cast<size_t>(n + 1) * K_weight,
                B + static_cast<size_t>(n + 2) * K_weight,
                B + static_cast<size_t>(n + 3) * K_weight};
            __m256 acc0[kDenseNTile] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps()};
            __m256 acc1[kDenseNTile] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps()};
            int k = 0;
            for (; k + 15 < K; k += 16) {
                const __m256 a_lo = _mm256_loadu_ps(a_row + k);
                const __m256 a_hi = _mm256_loadu_ps(a_row + k + 8);
                for (int t = 0; t < kDenseNTile; ++t) {
                    acc0[t] = _mm256_fmadd_ps(
                        a_lo, _mm256_loadu_ps(b_rows[t] + k), acc0[t]);
                    acc1[t] = _mm256_fmadd_ps(
                        a_hi, _mm256_loadu_ps(b_rows[t] + k + 8), acc1[t]);
                }
            }
            __m256 acc[kDenseNTile];
            for (int t = 0; t < kDenseNTile; ++t)
                acc[t] = _mm256_add_ps(acc0[t], acc1[t]);
            for (; k + 7 < K; k += 8) {
                const __m256 a = _mm256_loadu_ps(a_row + k);
                for (int t = 0; t < kDenseNTile; ++t) {
                    acc[t] = _mm256_fmadd_ps(
                        a, _mm256_loadu_ps(b_rows[t] + k), acc[t]);
                }
            }
            for (int t = 0; t < kDenseNTile; ++t) {
                float sum = horizontal_sum(acc[t]);
                for (int kk = k; kk < K; ++kk)
                    sum += a_row[kk] * b_rows[t][kk];
                c_row[n + t] = sum;
            }
        }
        for (; n < N; ++n) {
            const float* b_row = B + static_cast<size_t>(n) * K_weight;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 15 < K; k += 16) {
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                       _mm256_loadu_ps(b_row + k), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k + 8),
                                       _mm256_loadu_ps(b_row + k + 8), acc1);
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            for (; k + 7 < K; k += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                      _mm256_loadu_ps(b_row + k), acc);
            }
            float sum = horizontal_sum(acc);
            for (; k < K; ++k)
                sum += a_row[k] * b_row[k];
            c_row[n] = sum;
        }
    }
}

void matmul_fp16_avx2_range(const float* A, const fp16_t* B, float* C, int N,
                            int K, int lda, int K_weight, int ldc,
                            int m_begin, int m_end) {
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = A + static_cast<size_t>(m) * lda;
        float* c_row = C + static_cast<size_t>(m) * ldc;
        int n = 0;
        for (; n + kDenseNTile <= N; n += kDenseNTile) {
            const fp16_t* b_rows[kDenseNTile] = {
                B + static_cast<size_t>(n) * K_weight,
                B + static_cast<size_t>(n + 1) * K_weight,
                B + static_cast<size_t>(n + 2) * K_weight,
                B + static_cast<size_t>(n + 3) * K_weight};
            __m256 acc0[kDenseNTile] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps()};
            __m256 acc1[kDenseNTile] = {
                _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(),
                _mm256_setzero_ps()};
            int k = 0;
            for (; k + 15 < K; k += 16) {
                const __m256 a_lo = _mm256_loadu_ps(a_row + k);
                const __m256 a_hi = _mm256_loadu_ps(a_row + k + 8);
                for (int t = 0; t < kDenseNTile; ++t) {
                    const __m128i b0 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(b_rows[t] + k));
                    const __m128i b1 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(b_rows[t] + k + 8));
                    acc0[t] = _mm256_fmadd_ps(a_lo, _mm256_cvtph_ps(b0),
                                              acc0[t]);
                    acc1[t] = _mm256_fmadd_ps(a_hi, _mm256_cvtph_ps(b1),
                                              acc1[t]);
                }
            }
            __m256 acc[kDenseNTile];
            for (int t = 0; t < kDenseNTile; ++t)
                acc[t] = _mm256_add_ps(acc0[t], acc1[t]);
            for (; k + 7 < K; k += 8) {
                const __m256 a = _mm256_loadu_ps(a_row + k);
                for (int t = 0; t < kDenseNTile; ++t) {
                    const __m128i packed = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(b_rows[t] + k));
                    acc[t] = _mm256_fmadd_ps(a, _mm256_cvtph_ps(packed),
                                             acc[t]);
                }
            }
            for (int t = 0; t < kDenseNTile; ++t) {
                float sum = horizontal_sum(acc[t]);
                for (int kk = k; kk < K; ++kk)
                    sum += a_row[kk] * static_cast<float>(b_rows[t][kk]);
                c_row[n + t] = sum;
            }
        }
        for (; n < N; ++n) {
            const fp16_t* b_row = B + static_cast<size_t>(n) * K_weight;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            int k = 0;
            for (; k + 15 < K; k += 16) {
                const __m128i b0 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_row + k));
                const __m128i b1 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_row + k + 8));
                acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                       _mm256_cvtph_ps(b0), acc0);
                acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k + 8),
                                       _mm256_cvtph_ps(b1), acc1);
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            for (; k + 7 < K; k += 8) {
                const __m128i packed = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(b_row + k));
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(a_row + k),
                                      _mm256_cvtph_ps(packed), acc);
            }
            float sum = horizontal_sum(acc);
            for (; k < K; ++k)
                sum += a_row[k] * static_cast<float>(b_row[k]);
            c_row[n] = sum;
        }
    }
}

void matmul_int4_bg_avx2_range(const Tensor& A, const Tensor& B, Tensor& C,
                               int lda, int ldc, int m_begin, int m_end,
                               int n_begin, int n_end) {
    const int groups = static_cast<int>(B.groups_per_row);
    const auto* bg32 = static_cast<const Q4B8G32Block*>(B.q4_g32_data);
    const auto* bg128 = static_cast<const Q4B8G128Block*>(B.q4_g128_data);
    const bool is_bg32 = B.is_q4_g32_packed && bg32 && B.group_size == 32;

    if (m_end - m_begin == 1) {
        const float* a_row =
            A.ptr<float>() + static_cast<size_t>(m_begin) * lda;
        float* out =
            C.ptr<float>() + static_cast<size_t>(m_begin) * ldc;
        if (is_bg32) {
            run_n_tiles(
                n_begin, n_end,
                [&](int n) { gemv_bg32_one_n(a_row, out, n, bg32, groups); },
                [&](int n) { gemv_bg32_tile8(a_row, out, n, bg32, groups); });
        } else {
            run_n_tiles(
                n_begin, n_end,
                [&](int n) {
                    gemv_bg128_one_n(a_row, out, n, bg128, groups);
                },
                [&](int n) {
                    gemv_bg128_tile8(a_row, out, n, bg128, groups);
                });
        }
        return;
    }

    for (int m_base = m_begin; m_base < m_end; m_base += kMTile) {
        const int m_count = std::min(kMTile, m_end - m_base);
        for (int n = n_begin; n < n_end; ++n) {
            if (is_bg32) {
                gemm_bg32_one_n(A, C, lda, ldc, m_base, m_count, n, bg32,
                                groups);
            } else {
                gemm_bg128_one_n(A, C, lda, ldc, m_base, m_count, n, bg128,
                                 groups);
            }
        }
    }
}

void matmul_int8_avx2_range(const float* A, const int8_t* B,
                            const float* scales, float* C, int N, int K,
                            int group_size, int groups_per_row, int lda,
                            int K_weight, int ldc, int m_begin, int m_end,
                            int n_begin, int n_end) {
    (void)N;
    for (int m = m_begin; m < m_end; ++m) {
        const float* a_row = A + static_cast<size_t>(m) * lda;
        float* c_row = C + static_cast<size_t>(m) * ldc;
        for (int n = n_begin; n < n_end; ++n) {
            const int8_t* b_row = B + static_cast<size_t>(n) * K_weight;
            const float* scale_row =
                scales + static_cast<size_t>(n) * groups_per_row;
            float sum = 0.0f;
            for (int group = 0; group < groups_per_row; ++group) {
                const int k_begin = group * group_size;
                const int k_end = std::min(k_begin + group_size, K);
                __m256 acc = _mm256_setzero_ps();
                int k = k_begin;
                for (; k + 7 < k_end; k += 8) {
                    const __m128i q = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(b_row + k));
                    const __m256 qf = _mm256_cvtepi32_ps(
                        _mm256_cvtepi8_epi32(q));
                    acc = _mm256_fmadd_ps(
                        _mm256_loadu_ps(a_row + k), qf, acc);
                }
                float group_sum = horizontal_sum(acc);
                for (; k < k_end; ++k)
                    group_sum += a_row[k] * static_cast<float>(b_row[k]);
                sum += group_sum * scale_row[group];
            }
            c_row[n] = sum;
        }
    }
}

namespace {

inline float horizontal_max(__m256 value) {
    const __m128 low = _mm256_castps256_ps128(value);
    const __m128 high = _mm256_extractf128_ps(value, 1);
    __m128 max = _mm_max_ps(low, high);
    max = _mm_max_ps(max, _mm_movehdup_ps(max));
    max = _mm_max_ps(max, _mm_movehl_ps(max, max));
    return _mm_cvtss_f32(max);
}

inline __m256i pack_32_epi32_to_epi8(__m256i a, __m256i b, __m256i c,
                                     __m256i d) {
    const __m256i ab = _mm256_packs_epi32(a, b);
    const __m256i cd = _mm256_packs_epi32(c, d);
    return _mm256_permutevar8x32_epi32(
        _mm256_packs_epi16(ab, cd),
        _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

inline __m256i quantize_32_q8(const float* source, __m256 inverse) {
    const __m256 lo = _mm256_set1_ps(-127.0f);
    const __m256 hi = _mm256_set1_ps(127.0f);
    const __m256i i0 = _mm256_cvtps_epi32(_mm256_min_ps(
        hi, _mm256_max_ps(lo, _mm256_mul_ps(_mm256_loadu_ps(source), inverse))));
    const __m256i i1 = _mm256_cvtps_epi32(_mm256_min_ps(
        hi, _mm256_max_ps(
                lo, _mm256_mul_ps(_mm256_loadu_ps(source + 8), inverse))));
    const __m256i i2 = _mm256_cvtps_epi32(_mm256_min_ps(
        hi, _mm256_max_ps(
                lo, _mm256_mul_ps(_mm256_loadu_ps(source + 16), inverse))));
    const __m256i i3 = _mm256_cvtps_epi32(_mm256_min_ps(
        hi, _mm256_max_ps(
                lo, _mm256_mul_ps(_mm256_loadu_ps(source + 24), inverse))));
    return pack_32_epi32_to_epi8(i0, i1, i2, i3);
}

inline int32_t hsum_epi8(__m256i values) {
    const __m256i pair_sums =
        _mm256_maddubs_epi16(_mm256_set1_epi8(1), values);
    const __m256i quad_sums =
        _mm256_madd_epi16(pair_sums, _mm256_set1_epi16(1));
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(quad_sums),
                                _mm256_extracti128_si256(quad_sums, 1));
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(sum);
}

inline __m256i epi8_even_odd(__m256i sequential) {
    const __m256i control = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    const __m256i shuffled = _mm256_shuffle_epi8(sequential, control);
    const __m128i low = _mm256_castsi256_si128(shuffled);
    const __m128i high = _mm256_extracti128_si256(shuffled, 1);
    return _mm256_set_m128i(_mm_unpackhi_epi64(low, high),
                            _mm_unpacklo_epi64(low, high));
}

inline int32_t hsum_epi32(__m256i value) {
    __m128i sum = _mm_add_epi32(_mm256_castsi256_si128(value),
                                _mm256_extracti128_si256(value, 1));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2)));
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(sum);
}

// Reduce 8 YMM int32 accumulators into [hsum(a0), ..., hsum(a7)].
inline __m256i hsum_8x_epi32(__m256i a0, __m256i a1, __m256i a2, __m256i a3,
                             __m256i a4, __m256i a5, __m256i a6, __m256i a7) {
    const __m256i t0 = _mm256_hadd_epi32(a0, a1);
    const __m256i t1 = _mm256_hadd_epi32(a2, a3);
    const __m256i t2 = _mm256_hadd_epi32(a4, a5);
    const __m256i t3 = _mm256_hadd_epi32(a6, a7);
    const __m256i t4 = _mm256_hadd_epi32(t0, t1);
    const __m256i t5 = _mm256_hadd_epi32(t2, t3);
    return _mm256_add_epi32(_mm256_permute2x128_si256(t4, t5, 0x20),
                            _mm256_permute2x128_si256(t4, t5, 0x31));
}

inline void prefetch_bytes(const void* ptr, int bytes) {
    const char* p = static_cast<const char*>(ptr);
    for (int offset = 0; offset < bytes; offset += 64)
        _mm_prefetch(p + offset, _MM_HINT_T0);
}

inline __m256i q4_even_odd_plus8(const uint8_t* packed, __m256i nibble_mask,
                                 __m256i plus8) {
    const __m128i raw =
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    return _mm256_xor_si256(
        _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw),
                         nibble_mask),
        plus8);
}

inline __m256i madd_q8_w(__m256i weights, __m256i activations,
                         __m256i ones_epi16) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(weights, activations),
                             ones_epi16);
}

inline __m256i madd_q4_q8_even_odd(const uint8_t* packed, __m256i activations,
                                   __m256i nibble_mask, __m256i plus8,
                                   __m256i ones_epi16) {
    return madd_q8_w(q4_even_odd_plus8(packed, nibble_mask, plus8), activations,
                     ones_epi16);
}

inline void load_q4_w8(const uint8_t* packed, __m256i nibble_mask,
                       __m256i plus8, __m256i w[8]) {
    w[0] = q4_even_odd_plus8(packed, nibble_mask, plus8);
    w[1] = q4_even_odd_plus8(packed + 16, nibble_mask, plus8);
    w[2] = q4_even_odd_plus8(packed + 32, nibble_mask, plus8);
    w[3] = q4_even_odd_plus8(packed + 48, nibble_mask, plus8);
    w[4] = q4_even_odd_plus8(packed + 64, nibble_mask, plus8);
    w[5] = q4_even_odd_plus8(packed + 80, nibble_mask, plus8);
    w[6] = q4_even_odd_plus8(packed + 96, nibble_mask, plus8);
    w[7] = q4_even_odd_plus8(packed + 112, nibble_mask, plus8);
}

inline void madd_w8_row(const __m256i w[8], __m256i activations,
                        __m256i ones, __m256i acc[8]) {
    acc[0] = _mm256_add_epi32(acc[0], madd_q8_w(w[0], activations, ones));
    acc[1] = _mm256_add_epi32(acc[1], madd_q8_w(w[1], activations, ones));
    acc[2] = _mm256_add_epi32(acc[2], madd_q8_w(w[2], activations, ones));
    acc[3] = _mm256_add_epi32(acc[3], madd_q8_w(w[3], activations, ones));
    acc[4] = _mm256_add_epi32(acc[4], madd_q8_w(w[4], activations, ones));
    acc[5] = _mm256_add_epi32(acc[5], madd_q8_w(w[5], activations, ones));
    acc[6] = _mm256_add_epi32(acc[6], madd_q8_w(w[6], activations, ones));
    acc[7] = _mm256_add_epi32(acc[7], madd_q8_w(w[7], activations, ones));
}

inline void zero_acc8(__m256i acc[8]) {
    acc[0] = _mm256_setzero_si256();
    acc[1] = _mm256_setzero_si256();
    acc[2] = _mm256_setzero_si256();
    acc[3] = _mm256_setzero_si256();
    acc[4] = _mm256_setzero_si256();
    acc[5] = _mm256_setzero_si256();
    acc[6] = _mm256_setzero_si256();
    acc[7] = _mm256_setzero_si256();
}

inline __m256 scale_row8(const __m256i acc[8], int32_t bias, float a_scale,
                         __m256 b_scales, __m256 fsum) {
    const __m256i dots = _mm256_sub_epi32(
        hsum_8x_epi32(acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6],
                      acc[7]),
        _mm256_set1_epi32(bias));
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(dots),
                           _mm256_mul_ps(_mm256_set1_ps(a_scale), b_scales),
                           fsum);
}

inline void q8_gemv_bg32(const int8_t* qA, const float* a_scales,
                         const int32_t* a_sums, float* out,
                         const Q4B8G32Block* bg32, int groups, int n_begin,
                         int n_end) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256i plus8 = _mm256_set1_epi8(0x08);
    auto one_n = [&](int n) {
        float sum = 0.0f;
        const int lane = n & 7;
        const size_t tile = static_cast<size_t>(n / 8);
        for (int group = 0; group < groups; ++group) {
            const auto& block = bg32[tile * groups + group];
            const __m256i activations = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    qA + static_cast<size_t>(group) * 32));
            const int32_t dot =
                hsum_epi32(madd_q4_q8_even_odd(block.q[lane], activations,
                                               nibble_mask, plus8, ones)) -
                8 * a_sums[group];
            sum += static_cast<float>(dot) * a_scales[group] *
                   block.scales[lane];
        }
        out[n] = sum;
    };
    auto tile8 = [&](int n) {
        __m256 fsum = _mm256_setzero_ps();
        const size_t tile = static_cast<size_t>(n / 8);
        for (int group = 0; group < groups; ++group) {
            const auto& block = bg32[tile * groups + group];
            if (group + 1 < groups)
                prefetch_bytes(&bg32[tile * groups + group + 1],
                               static_cast<int>(sizeof(Q4B8G32Block)));
            const __m256i activations = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(
                    qA + static_cast<size_t>(group) * 32));
            const uint8_t* packed = &block.q[0][0];
            const __m256i dots = _mm256_sub_epi32(
                hsum_8x_epi32(
                    madd_q4_q8_even_odd(packed, activations, nibble_mask, plus8,
                                        ones),
                    madd_q4_q8_even_odd(packed + 16, activations, nibble_mask,
                                        plus8, ones),
                    madd_q4_q8_even_odd(packed + 32, activations, nibble_mask,
                                        plus8, ones),
                    madd_q4_q8_even_odd(packed + 48, activations, nibble_mask,
                                        plus8, ones),
                    madd_q4_q8_even_odd(packed + 64, activations, nibble_mask,
                                        plus8, ones),
                    madd_q4_q8_even_odd(packed + 80, activations, nibble_mask,
                                        plus8, ones),
                    madd_q4_q8_even_odd(packed + 96, activations, nibble_mask,
                                        plus8, ones),
                    madd_q4_q8_even_odd(packed + 112, activations, nibble_mask,
                                        plus8, ones)),
                _mm256_set1_epi32(8 * a_sums[group]));
            fsum = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(dots),
                _mm256_mul_ps(_mm256_set1_ps(a_scales[group]),
                              _mm256_loadu_ps(block.scales)),
                fsum);
        }
        _mm256_storeu_ps(out + n, fsum);
    };
    run_n_tiles(n_begin, n_end, one_n, tile8);
}

inline void q8_gemv_bg128(const int8_t* qA, const float* a_scales,
                          const int32_t* a_sums, float* out,
                          const Q4B8G128Block* bg128, int groups, int n_begin,
                          int n_end) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256i plus8 = _mm256_set1_epi8(0x08);
    auto one_n = [&](int n) {
        float sum = 0.0f;
        const int lane = n & 7;
        const size_t tile = static_cast<size_t>(n / 8);
        for (int group = 0; group < groups; ++group) {
            const auto& block = bg128[tile * groups + group];
            const int8_t* a_group = qA + static_cast<size_t>(group) * 128;
            __m256i acc = _mm256_setzero_si256();
            for (int sub = 0; sub < 4; ++sub) {
                const __m256i activations = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_group + sub * 32));
                acc = _mm256_add_epi32(
                    acc, madd_q4_q8_even_odd(block.q[sub][lane], activations,
                                             nibble_mask, plus8, ones));
            }
            const int32_t dot = hsum_epi32(acc) - 8 * a_sums[group];
            sum += static_cast<float>(dot) * a_scales[group] *
                   block.scales[lane];
        }
        out[n] = sum;
    };
    auto tile8 = [&](int n) {
        __m256 fsum = _mm256_setzero_ps();
        const size_t tile = static_cast<size_t>(n / 8);
        if (n + 8 < n_end)
            prefetch_bytes(&bg128[(tile + 1) * groups],
                           static_cast<int>(sizeof(Q4B8G128Block)));
        for (int group = 0; group < groups; ++group) {
            const auto& block = bg128[tile * groups + group];
            if (group + 1 < groups)
                prefetch_bytes(&bg128[tile * groups + group + 1],
                               static_cast<int>(sizeof(Q4B8G128Block)));
            const int8_t* a_group = qA + static_cast<size_t>(group) * 128;
            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();
            __m256i acc4 = _mm256_setzero_si256();
            __m256i acc5 = _mm256_setzero_si256();
            __m256i acc6 = _mm256_setzero_si256();
            __m256i acc7 = _mm256_setzero_si256();
            for (int sub = 0; sub < 4; ++sub) {
                const __m256i activations = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_group + sub * 32));
                const uint8_t* packed = &block.q[sub][0][0];
                acc0 = _mm256_add_epi32(
                    acc0, madd_q4_q8_even_odd(packed, activations, nibble_mask,
                                              plus8, ones));
                acc1 = _mm256_add_epi32(
                    acc1, madd_q4_q8_even_odd(packed + 16, activations,
                                              nibble_mask, plus8, ones));
                acc2 = _mm256_add_epi32(
                    acc2, madd_q4_q8_even_odd(packed + 32, activations,
                                              nibble_mask, plus8, ones));
                acc3 = _mm256_add_epi32(
                    acc3, madd_q4_q8_even_odd(packed + 48, activations,
                                              nibble_mask, plus8, ones));
                acc4 = _mm256_add_epi32(
                    acc4, madd_q4_q8_even_odd(packed + 64, activations,
                                              nibble_mask, plus8, ones));
                acc5 = _mm256_add_epi32(
                    acc5, madd_q4_q8_even_odd(packed + 80, activations,
                                              nibble_mask, plus8, ones));
                acc6 = _mm256_add_epi32(
                    acc6, madd_q4_q8_even_odd(packed + 96, activations,
                                              nibble_mask, plus8, ones));
                acc7 = _mm256_add_epi32(
                    acc7, madd_q4_q8_even_odd(packed + 112, activations,
                                              nibble_mask, plus8, ones));
            }
            const __m256i dots = _mm256_sub_epi32(
                hsum_8x_epi32(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7),
                _mm256_set1_epi32(8 * a_sums[group]));
            fsum = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(dots),
                _mm256_mul_ps(_mm256_set1_ps(a_scales[group]),
                              _mm256_loadu_ps(block.scales)),
                fsum);
        }
        _mm256_storeu_ps(out + n, fsum);
    };
    run_n_tiles(n_begin, n_end, one_n, tile8);
}

inline void q8_gemm_bg32(const int8_t* qA, const float* a_scales,
                         const int32_t* a_sums, float* C, int M, int K,
                         int ldc, const Q4B8G32Block* bg32, int groups,
                         int n_begin, int n_end) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256i plus8 = _mm256_set1_epi8(0x08);
    auto one_n = [&](int n) {
        const int lane = n & 7;
        const size_t tile = static_cast<size_t>(n / 8);
        for (int m = 0; m < M; ++m) {
            float sum = 0.0f;
            const int8_t* a_row = qA + static_cast<size_t>(m) * K;
            const float* scales = a_scales + static_cast<size_t>(m) * groups;
            const int32_t* sums = a_sums + static_cast<size_t>(m) * groups;
            for (int group = 0; group < groups; ++group) {
                const auto& block = bg32[tile * groups + group];
                const __m256i activations = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(a_row + group * 32));
                const int32_t dot =
                    hsum_epi32(madd_q4_q8_even_odd(block.q[lane], activations,
                                                   nibble_mask, plus8, ones)) -
                    8 * sums[group];
                sum += static_cast<float>(dot) * scales[group] *
                       block.scales[lane];
            }
            C[static_cast<size_t>(m) * ldc + n] = sum;
        }
    };
    auto tile8 = [&](int n) {
        const size_t tile = static_cast<size_t>(n / 8);
        int m = 0;
        for (; m + 4 <= M; m += 4) {
            __m256 fsum0 = _mm256_setzero_ps();
            __m256 fsum1 = _mm256_setzero_ps();
            __m256 fsum2 = _mm256_setzero_ps();
            __m256 fsum3 = _mm256_setzero_ps();
            const int8_t* a0 = qA + static_cast<size_t>(m) * K;
            const int8_t* a1 = a0 + K;
            const int8_t* a2 = a1 + K;
            const int8_t* a3 = a2 + K;
            const float* s0 = a_scales + static_cast<size_t>(m) * groups;
            const float* s1 = s0 + groups;
            const float* s2 = s1 + groups;
            const float* s3 = s2 + groups;
            const int32_t* u0 = a_sums + static_cast<size_t>(m) * groups;
            const int32_t* u1 = u0 + groups;
            const int32_t* u2 = u1 + groups;
            const int32_t* u3 = u2 + groups;
            for (int group = 0; group < groups; ++group) {
                const auto& block = bg32[tile * groups + group];
                if (group + 1 < groups)
                    prefetch_bytes(&bg32[tile * groups + group + 1],
                                   static_cast<int>(sizeof(Q4B8G32Block)));
                __m256i w[8];
                load_q4_w8(&block.q[0][0], nibble_mask, plus8, w);
                __m256i acc0[8], acc1[8], acc2[8], acc3[8];
                zero_acc8(acc0);
                zero_acc8(acc1);
                zero_acc8(acc2);
                zero_acc8(acc3);
                madd_w8_row(w,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                a0 + group * 32)),
                            ones, acc0);
                madd_w8_row(w,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                a1 + group * 32)),
                            ones, acc1);
                madd_w8_row(w,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                a2 + group * 32)),
                            ones, acc2);
                madd_w8_row(w,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                a3 + group * 32)),
                            ones, acc3);
                const __m256 b_scales = _mm256_loadu_ps(block.scales);
                fsum0 = scale_row8(acc0, 8 * u0[group], s0[group], b_scales,
                                   fsum0);
                fsum1 = scale_row8(acc1, 8 * u1[group], s1[group], b_scales,
                                   fsum1);
                fsum2 = scale_row8(acc2, 8 * u2[group], s2[group], b_scales,
                                   fsum2);
                fsum3 = scale_row8(acc3, 8 * u3[group], s3[group], b_scales,
                                   fsum3);
            }
            _mm256_storeu_ps(C + static_cast<size_t>(m) * ldc + n, fsum0);
            _mm256_storeu_ps(C + static_cast<size_t>(m + 1) * ldc + n, fsum1);
            _mm256_storeu_ps(C + static_cast<size_t>(m + 2) * ldc + n, fsum2);
            _mm256_storeu_ps(C + static_cast<size_t>(m + 3) * ldc + n, fsum3);
        }
        for (; m < M; ++m) {
            __m256 fsum = _mm256_setzero_ps();
            const int8_t* a_row = qA + static_cast<size_t>(m) * K;
            const float* scales = a_scales + static_cast<size_t>(m) * groups;
            const int32_t* sums = a_sums + static_cast<size_t>(m) * groups;
            for (int group = 0; group < groups; ++group) {
                const auto& block = bg32[tile * groups + group];
                __m256i w[8];
                load_q4_w8(&block.q[0][0], nibble_mask, plus8, w);
                __m256i acc[8];
                zero_acc8(acc);
                madd_w8_row(w,
                            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                                a_row + group * 32)),
                            ones, acc);
                fsum = scale_row8(acc, 8 * sums[group], scales[group],
                                  _mm256_loadu_ps(block.scales), fsum);
            }
            _mm256_storeu_ps(C + static_cast<size_t>(m) * ldc + n, fsum);
        }
    };
    run_n_tiles(n_begin, n_end, one_n, tile8);
}

inline void q8_gemm_bg128(const int8_t* qA, const float* a_scales,
                          const int32_t* a_sums, float* C, int M, int K,
                          int ldc, const Q4B8G128Block* bg128, int groups,
                          int n_begin, int n_end) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
    const __m256i plus8 = _mm256_set1_epi8(0x08);
    auto one_n = [&](int n) {
        const int lane = n & 7;
        const size_t tile = static_cast<size_t>(n / 8);
        for (int m = 0; m < M; ++m) {
            float sum = 0.0f;
            const int8_t* a_row = qA + static_cast<size_t>(m) * K;
            const float* scales = a_scales + static_cast<size_t>(m) * groups;
            const int32_t* sums = a_sums + static_cast<size_t>(m) * groups;
            for (int group = 0; group < groups; ++group) {
                const auto& block = bg128[tile * groups + group];
                const int8_t* a_group = a_row + static_cast<size_t>(group) * 128;
                __m256i acc = _mm256_setzero_si256();
                for (int sub = 0; sub < 4; ++sub) {
                    const __m256i activations = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(a_group + sub * 32));
                    acc = _mm256_add_epi32(
                        acc, madd_q4_q8_even_odd(block.q[sub][lane],
                                                 activations, nibble_mask,
                                                 plus8, ones));
                }
                const int32_t dot = hsum_epi32(acc) - 8 * sums[group];
                sum += static_cast<float>(dot) * scales[group] *
                       block.scales[lane];
            }
            C[static_cast<size_t>(m) * ldc + n] = sum;
        }
    };
    auto tile8 = [&](int n) {
        const size_t tile = static_cast<size_t>(n / 8);
        if (n + 8 < n_end)
            prefetch_bytes(&bg128[(tile + 1) * groups],
                           static_cast<int>(sizeof(Q4B8G128Block)));
        int m = 0;
        for (; m + 4 <= M; m += 4) {
            __m256 fsum0 = _mm256_setzero_ps();
            __m256 fsum1 = _mm256_setzero_ps();
            __m256 fsum2 = _mm256_setzero_ps();
            __m256 fsum3 = _mm256_setzero_ps();
            const int8_t* a0 = qA + static_cast<size_t>(m) * K;
            const int8_t* a1 = a0 + K;
            const int8_t* a2 = a1 + K;
            const int8_t* a3 = a2 + K;
            const float* s0 = a_scales + static_cast<size_t>(m) * groups;
            const float* s1 = s0 + groups;
            const float* s2 = s1 + groups;
            const float* s3 = s2 + groups;
            const int32_t* u0 = a_sums + static_cast<size_t>(m) * groups;
            const int32_t* u1 = u0 + groups;
            const int32_t* u2 = u1 + groups;
            const int32_t* u3 = u2 + groups;
            for (int group = 0; group < groups; ++group) {
                const auto& block = bg128[tile * groups + group];
                if (group + 1 < groups)
                    prefetch_bytes(&bg128[tile * groups + group + 1],
                                   static_cast<int>(sizeof(Q4B8G128Block)));
                __m256i acc0[8], acc1[8], acc2[8], acc3[8];
                zero_acc8(acc0);
                zero_acc8(acc1);
                zero_acc8(acc2);
                zero_acc8(acc3);
                for (int sub = 0; sub < 4; ++sub) {
                    __m256i w[8];
                    load_q4_w8(&block.q[sub][0][0], nibble_mask, plus8, w);
                    const int off = group * 128 + sub * 32;
                    madd_w8_row(w,
                                _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i*>(a0 + off)),
                                ones, acc0);
                    madd_w8_row(w,
                                _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i*>(a1 + off)),
                                ones, acc1);
                    madd_w8_row(w,
                                _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i*>(a2 + off)),
                                ones, acc2);
                    madd_w8_row(w,
                                _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i*>(a3 + off)),
                                ones, acc3);
                }
                const __m256 b_scales = _mm256_loadu_ps(block.scales);
                fsum0 = scale_row8(acc0, 8 * u0[group], s0[group], b_scales,
                                   fsum0);
                fsum1 = scale_row8(acc1, 8 * u1[group], s1[group], b_scales,
                                   fsum1);
                fsum2 = scale_row8(acc2, 8 * u2[group], s2[group], b_scales,
                                   fsum2);
                fsum3 = scale_row8(acc3, 8 * u3[group], s3[group], b_scales,
                                   fsum3);
            }
            _mm256_storeu_ps(C + static_cast<size_t>(m) * ldc + n, fsum0);
            _mm256_storeu_ps(C + static_cast<size_t>(m + 1) * ldc + n, fsum1);
            _mm256_storeu_ps(C + static_cast<size_t>(m + 2) * ldc + n, fsum2);
            _mm256_storeu_ps(C + static_cast<size_t>(m + 3) * ldc + n, fsum3);
        }
        for (; m < M; ++m) {
            __m256 fsum = _mm256_setzero_ps();
            const int8_t* a_row = qA + static_cast<size_t>(m) * K;
            const float* scales = a_scales + static_cast<size_t>(m) * groups;
            const int32_t* sums = a_sums + static_cast<size_t>(m) * groups;
            for (int group = 0; group < groups; ++group) {
                const auto& block = bg128[tile * groups + group];
                __m256i acc[8];
                zero_acc8(acc);
                for (int sub = 0; sub < 4; ++sub) {
                    __m256i w[8];
                    load_q4_w8(&block.q[sub][0][0], nibble_mask, plus8, w);
                    madd_w8_row(w,
                                _mm256_loadu_si256(
                                    reinterpret_cast<const __m256i*>(
                                        a_row + group * 128 + sub * 32)),
                                ones, acc);
                }
                fsum = scale_row8(acc, 8 * sums[group], scales[group],
                                  _mm256_loadu_ps(block.scales), fsum);
            }
            _mm256_storeu_ps(C + static_cast<size_t>(m) * ldc + n, fsum);
        }
    };
    run_n_tiles(n_begin, n_end, one_n, tile8);
}

}  // namespace

void quantize_q8_avx2(const float* A, int8_t* qA, float* scales, int32_t* sums,
                      int K, int group_size) {
    const int groups = K / group_size;
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (int group = 0; group < groups; ++group) {
        const float* source = A + static_cast<size_t>(group) * group_size;
        int8_t* output = qA + static_cast<size_t>(group) * group_size;
        __m256 amax = _mm256_setzero_ps();
        for (int k = 0; k < group_size; k += 8) {
            amax = _mm256_max_ps(
                amax, _mm256_andnot_ps(sign_mask, _mm256_loadu_ps(source + k)));
        }
        const float peak = horizontal_max(amax);
        const float scale = peak > 0.0f ? peak / 127.0f : 1.0f;
        const float inverse = peak > 0.0f ? 127.0f / peak : 0.0f;
        scales[group] = scale;
        const __m256 inverse_v = _mm256_set1_ps(inverse);
        int32_t sum = 0;
        for (int k = 0; k < group_size; k += 32) {
            const __m256i quantized = quantize_32_q8(source + k, inverse_v);
            sum += hsum_epi8(quantized);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + k),
                                epi8_even_odd(quantized));
        }
        sums[group] = sum;
    }
}

void matmul_int4_q8_bg_avx2_gemv(const int8_t* qA, const float* a_scales,
                                 const int32_t* a_sums, const Tensor& B,
                                 Tensor& C, int n_begin, int n_end) {
    float* out = C.ptr<float>();
    const int groups = static_cast<int>(B.groups_per_row);
    if (B.is_q4_g32_packed && B.q4_g32_data && B.group_size == 32) {
        q8_gemv_bg32(qA, a_scales, a_sums, out,
                     static_cast<const Q4B8G32Block*>(B.q4_g32_data), groups,
                     n_begin, n_end);
        return;
    }
    q8_gemv_bg128(qA, a_scales, a_sums, out,
                  static_cast<const Q4B8G128Block*>(B.q4_g128_data), groups,
                  n_begin, n_end);
}

void matmul_int4_q8_bg_avx2_gemm(const int8_t* qA, const float* a_scales,
                                 const int32_t* a_sums, const Tensor& B,
                                 Tensor& C, int M, int K, int ldc, int n_begin,
                                 int n_end) {
    float* out = C.ptr<float>();
    const int groups = static_cast<int>(B.groups_per_row);
    if (B.is_q4_g32_packed && B.q4_g32_data && B.group_size == 32) {
        q8_gemm_bg32(qA, a_scales, a_sums, out, M, K, ldc,
                     static_cast<const Q4B8G32Block*>(B.q4_g32_data), groups,
                     n_begin, n_end);
        return;
    }
    q8_gemm_bg128(qA, a_scales, a_sums, out, M, K, ldc,
                  static_cast<const Q4B8G128Block*>(B.q4_g128_data), groups,
                  n_begin, n_end);
}

}  // namespace mollm::cpu::x86
