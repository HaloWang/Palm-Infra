#pragma once

// CPU architecture boundary.
//
// Graph and engine code use this header for storage-level FP16 and for the
// few execution-policy decisions that genuinely vary by CPU family.  ARM
// intrinsics stay behind this boundary; generic kernels must not infer their
// target from compiler predefined macros.

#include <cstdint>

struct Tensor;
class ThreadPool;

#ifndef MOLLM_CPU_ARM_NEON
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
#define MOLLM_CPU_ARM_NEON 1
#else
#define MOLLM_CPU_ARM_NEON 0
#endif
#endif

#ifndef MOLLM_ARM_I8MM_KERNELS
#define MOLLM_ARM_I8MM_KERNELS 0
#endif

#if MOLLM_CPU_ARM_NEON
#include <arm_neon.h>
#endif

#if defined(_MSC_VER) && !MOLLM_CPU_ARM_NEON
#include <cstring>
#endif

namespace mollm::cpu {

enum class X86Isa : uint8_t {
    SCALAR = 0,
    AVX2 = 1,
    AVX512 = 2,
};

#if MOLLM_CPU_ARM_NEON
using fp16_t = __fp16;
#elif defined(__clang__)
// Clang exposes __fp16 as a storage-only type on x86. Keep it as the
// canonical storage spelling so legacy kernel signatures remain compatible.
using fp16_t = __fp16;
#elif defined(_MSC_VER)
// MSVC has no _Float16 / __fp16. Store IEEE binary16 bits and convert
// through float so existing `(__fp16)x` / `(float)h` call sites compile.
// Software conversion uses the IEEE default: round-to-nearest, ties-to-even,
// matching GCC `_Float16`. x86 F16C / WOA NEON can replace this later.
inline uint16_t fp16_bits_from_f32(float value) {
    auto round_rne = [](uint32_t magnitude, int shift) -> uint32_t {
        if (shift <= 0)
            return magnitude;
        if (shift >= 32)
            return 0;
        const uint32_t kept = magnitude >> shift;
        const uint32_t round_bit = (magnitude >> (shift - 1)) & 1u;
        const uint32_t sticky =
            shift == 1 ? 0u : (magnitude & ((1u << (shift - 1)) - 1u));
        if (round_bit != 0 && (sticky != 0 || (kept & 1u) != 0))
            return kept + 1u;
        return kept;
    };

    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
    const uint32_t abs_bits = bits & 0x7FFFFFFFu;
    if (abs_bits >= 0x7F800000u) {
        if (abs_bits == 0x7F800000u)
            return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(
            sign | 0x7E00u | ((abs_bits >> 13) & 0x1FFu));
    }

    const int exp = static_cast<int>(abs_bits >> 23);
    const uint32_t frac = abs_bits & 0x7FFFFFu;
    // f32 denormals are far below f16 min subnormal (2^-24).
    if (exp == 0)
        return sign;

    const int exp16 = exp - 127 + 15;
    if (exp16 >= 31)
        return static_cast<uint16_t>(sign | 0x7C00u);
    if (exp16 <= 0) {
        if (exp16 < -10)
            return sign;
        const uint32_t kept = round_rne(frac | 0x800000u, 14 - exp16);
        if (kept >= 0x400u)
            return static_cast<uint16_t>(sign | 0x0400u);
        return static_cast<uint16_t>(sign | kept);
    }

    const uint32_t kept = round_rne(frac, 13);
    int rounded_exp = exp16;
    uint32_t mantissa = kept;
    if (mantissa > 0x3FFu) {
        mantissa = 0;
        rounded_exp++;
    }
    if (rounded_exp >= 31)
        return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(
        sign | (static_cast<uint16_t>(rounded_exp) << 10) |
        static_cast<uint16_t>(mantissa));
}

inline float fp16_bits_to_f32(uint16_t value) {
    const uint16_t sign = static_cast<uint16_t>((value & 0x8000u) >> 15);
    const uint16_t exponent = static_cast<uint16_t>((value & 0x7C00u) >> 10);
    uint16_t significand = static_cast<uint16_t>(value & 0x03FFu);
    uint32_t bits = 0;
    if (exponent == 0) {
        if (significand == 0) {
            bits = static_cast<uint32_t>(sign) << 31;
        } else {
            int denorm_exp = 0;
            while ((significand & 0x200u) == 0) {
                significand = static_cast<uint16_t>(significand << 1);
                ++denorm_exp;
            }
            significand = static_cast<uint16_t>((significand << 1) & 0x3FFu);
            bits = (static_cast<uint32_t>(sign) << 31) |
                   (static_cast<uint32_t>(-denorm_exp - 15 + 127) << 23) |
                   (static_cast<uint32_t>(significand) << 13);
        }
    } else if (exponent == 0x1F) {
        bits = (static_cast<uint32_t>(sign) << 31) | (0xFFu << 23) |
               (static_cast<uint32_t>(significand) << 13);
    } else {
        bits = (static_cast<uint32_t>(sign) << 31) |
               (static_cast<uint32_t>(exponent - 15 + 127) << 23) |
               (static_cast<uint32_t>(significand) << 13);
    }
    float result = 0.f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct fp16_t {
    uint16_t bits = 0;

    fp16_t() = default;
    fp16_t(float value) noexcept : bits(fp16_bits_from_f32(value)) {}
    operator float() const noexcept { return fp16_bits_to_f32(bits); }
};
#else
// GCC supports IEEE binary16 storage on x86 Linux. It is used only for model
// bytes and scalar conversion; it does not imply native FP16 SIMD.
using fp16_t = _Float16;
#endif

static_assert(sizeof(fp16_t) == 2, "mollm FP16 storage must be binary16");

struct Capabilities {
    bool arm_neon = false;
    bool arm_i8mm = false;
    bool fp16_vector_math = false;
    bool fp16_kv_cache = false;
    bool fp16_interleaved_weights = false;
    bool x86_avx2 = false;
    bool x86_fma = false;
    bool x86_f16c = false;
    bool x86_avx512 = false;
    bool x86_avx512_vnni = false;
    X86Isa x86_isa = X86Isa::SCALAR;
};

const Capabilities& capabilities();
const char* isa_name();

// Hint while polling a CPU worker.  The ARM and scalar implementations live
// in separately selected translation units so no foreign assembly reaches a
// target compiler.
void relax();

// Handle a package-native packed INT4 matrix when the selected CPU provider
// has a portable decoder.  Returning false leaves the normal matmul dispatch
// to select its architecture-specific kernel.
bool matmul_int4_packed(const Tensor& A, const Tensor& B, Tensor& C, int lda,
                        int ldc, ThreadPool* thread_pool);

// Architecture-provider dense kernels. Returning false asks the caller to use
// the portable scalar implementation. The x86 provider binds these to
// separately compiled AVX2 or AVX-512 translation units after one runtime
// probe.
bool matmul_dense_fp32_range(const float* A, const float* B, float* C, int N,
                             int K, int lda, int K_weight, int ldc,
                             int m_begin, int m_end);
bool matmul_dense_fp16_range(const float* A, const fp16_t* B, float* C, int N,
                             int K, int lda, int K_weight, int ldc,
                             int m_begin, int m_end, bool interleaved);
bool matmul_int8_range(const float* A, const int8_t* B, const float* scales,
                       float* C, int N, int K, int group_size,
                       int groups_per_row, int lda, int K_weight, int ldc,
                       int m_begin, int m_end, int n_begin, int n_end,
                       bool interleaved);

}  // namespace mollm::cpu

#if !MOLLM_CPU_ARM_NEON && !defined(__clang__) && !defined(__CUDACC__)
// Legacy CPU kernels still spell their storage element as `__fp16`.  Keep the
// compatibility name at this one architecture boundary while those kernels
// are moved behind providers; no generic caller needs a compiler extension.
using __fp16 = mollm::cpu::fp16_t;
#endif

// Transitional compatibility for existing NEON kernels.  New generic code
// should use mollm::cpu::Capabilities instead of testing this macro.
#ifndef HAS_NEON
#define HAS_NEON MOLLM_CPU_ARM_NEON
#endif
