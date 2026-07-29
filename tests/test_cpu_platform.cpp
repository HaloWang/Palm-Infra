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
    if (caps.x86_isa == mollm::cpu::X86Isa::AVX2 &&
        (!caps.x86_avx2 || std::strcmp(name, "x86-avx2") != 0)) {
        std::fprintf(stderr, "inconsistent AVX2 dispatch state\n");
        return 1;
    }
    if (caps.x86_f16c && !caps.x86_avx2 && !caps.x86_avx512) {
        std::fprintf(stderr, "F16C selected without an x86 SIMD tier\n");
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

    std::printf("CPU provider: %s\n", name);
    return 0;
}
