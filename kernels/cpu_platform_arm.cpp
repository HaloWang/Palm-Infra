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

bool matmul_int4_packed(const Tensor&, const Tensor&, Tensor&, int, int,
                        ThreadPool*) {
    // ARM retains the upstream DOTPROD/i8mm dispatch in matmul_w4.cpp.
    return false;
}

bool matmul_dense_fp32_range(const float*, const float*, float*, int, int, int,
                             int, int, int, int) {
    return false;
}

bool matmul_dense_fp16_range(const float*, const fp16_t*, float*, int, int, int,
                             int, int, int, int, bool) {
    return false;
}

bool matmul_int8_range(const float*, const int8_t*, const float*, float*, int,
                       int, int, int, int, int, int, int, int, int, int, bool) {
    return false;
}

}  // namespace mollm::cpu
