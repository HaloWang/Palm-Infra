#include "kernels/cpu_platform.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
    const auto& caps = mollm::cpu::capabilities();
    const char* name = mollm::cpu::isa_name();
    if (!name || name[0] == '\0') {
        std::fprintf(stderr, "CPU provider name is empty\n");
        return 1;
    }

    if (caps.x86_avx512 &&
        (caps.x86_isa != mollm::cpu::X86Isa::AVX512 ||
         std::strcmp(name, "x86-avx512") != 0)) {
        std::fprintf(stderr, "inconsistent AVX-512 dispatch state\n");
        return 1;
    }
    if (caps.arm_i8mm &&
        (!caps.arm_neon || std::strcmp(name, "arm-neon-i8mm") != 0)) {
        std::fprintf(stderr, "inconsistent ARM i8mm dispatch state\n");
        return 1;
    }
    if (caps.x86_isa == mollm::cpu::X86Isa::AVX2 &&
        (!caps.x86_avx2 || std::strcmp(name, "x86-avx2") != 0)) {
        std::fprintf(stderr, "inconsistent AVX2 dispatch state\n");
        return 1;
    }
    if (caps.x86_f16c && !caps.x86_avx2 && !caps.x86_avx512) {
        std::fprintf(stderr, "F16C selected without an x86 SIMD tier\n");
        return 1;
    }
    if (caps.x86_avx512_vnni && !caps.x86_avx512) {
        std::fprintf(stderr, "AVX-512 VNNI selected without AVX-512\n");
        return 1;
    }

    const char* requested = std::getenv("MOLLM_X86_ISA");
    if ((std::getenv("MOLLM_X86_DISABLE_AVX2") ||
         (requested && std::strcmp(requested, "scalar") == 0)) &&
        std::strcmp(name, "x86-scalar") != 0) {
        std::fprintf(stderr, "scalar override selected %s\n", name);
        return 1;
    }
    if (requested && std::strcmp(requested, "avx2") == 0 &&
        caps.x86_avx512) {
        std::fprintf(stderr, "AVX2 cap incorrectly selected AVX-512\n");
        return 1;
    }
    const char* arm_requested = std::getenv("MOLLM_ARM_ISA");
    if ((std::getenv("MOLLM_ARM_DISABLE_I8MM") ||
         (arm_requested && std::strcmp(arm_requested, "neon") == 0)) &&
        caps.arm_i8mm) {
        std::fprintf(stderr, "ARM NEON cap incorrectly selected i8mm\n");
        return 1;
    }

    auto f32_from_bits = [](uint32_t bits) {
        float value = 0.f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    auto fp16_bits = [](float value) {
        const mollm::cpu::fp16_t half = value;
        uint16_t bits = 0;
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    };
    auto expect_fp16 = [&](float value, uint16_t expected, const char* label) {
        const uint16_t got = fp16_bits(value);
        if (got != expected) {
            std::fprintf(stderr, "%s: got 0x%04x expected 0x%04x\n", label,
                         got, expected);
            return false;
        }
        return true;
    };
    if (!expect_fp16(0.0f, 0x0000, "fp16 +0") ||
        !expect_fp16(-0.0f, 0x8000, "fp16 -0") ||
        !expect_fp16(1.0f, 0x3C00, "fp16 1.0") ||
        !expect_fp16(2.0f, 0x4000, "fp16 2.0") ||
        !expect_fp16(-2.0f, 0xC000, "fp16 -2.0") ||
        !expect_fp16(65504.0f, 0x7BFF, "fp16 max finite") ||
        !expect_fp16(65520.0f, 0x7C00, "fp16 overflow tie") ||
        !expect_fp16(f32_from_bits(0x3F801000u), 0x3C00,
                     "fp16 RNE tie-to-even") ||
        !expect_fp16(f32_from_bits(0x3F803000u), 0x3C02,
                     "fp16 RNE tie-from-odd")) {
        return 1;
    }

    std::printf("CPU provider: %s\n", name);
    return 0;
}
