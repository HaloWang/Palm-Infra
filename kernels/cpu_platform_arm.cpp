#include "kernels/cpu_platform.h"

namespace mollm::cpu {

const Capabilities& capabilities() {
    static const Capabilities value{/*arm_neon=*/true,
                                    /*fp16_vector_math=*/true,
                                    /*fp16_kv_cache=*/true,
                                    /*fp16_interleaved_weights=*/true};
    return value;
}

void relax() {
    __asm__ __volatile__("yield" ::: "memory");
}

bool matmul_int4_packed(const Tensor&, const Tensor&, Tensor&, int, int) {
    // ARM retains the upstream DOTPROD/i8mm dispatch in matmul_w4.cpp.
    return false;
}

}  // namespace mollm::cpu
