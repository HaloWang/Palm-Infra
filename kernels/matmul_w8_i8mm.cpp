// Compile the i8mm-only W8 implementation under a distinct object filename.
// Putting a second matmul_w8.cpp.o into the same static archive makes its
// archive member collide with the baseline object on the HarmonyOS LLVM
// toolchain, so the linker never discovers these runtime-dispatched symbols.
#define MOLLM_W8_I8MM_ONLY 1
#include "kernels/matmul_w8.cpp"
