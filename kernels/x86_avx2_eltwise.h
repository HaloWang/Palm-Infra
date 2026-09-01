#pragma once

void rms_norm_x86_avx2(const float* x, const float* weight, float* out, int D,
                       int N, float eps, int ldx, int ldo);
void add_rms_norm_row_x86_avx2(float* residual, const float* update, float* out,
                               const float* weight, int D, float eps);
void swiglu_row_x86_avx2(const float* gate, const float* up, float* out, int I);
void sigmoid_mul_row_x86_avx2(const float* value, const float* gate, float* out,
                              int D);
