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

}  // namespace mollm::cpu::x86
