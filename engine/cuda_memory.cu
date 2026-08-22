#include "engine/cuda_internal.h"

#include <cstdio>

namespace mollm_cuda {

bool report_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed: %s\n", operation,
                 cudaGetErrorString(error));
    return false;
}

bool report_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "CudaBackend: %s failed (cuBLAS status %d)\n",
                 operation, static_cast<int>(status));
    return false;
}

bool malloc_device(void** pointer, size_t bytes, const char* operation) {
    return report_cuda(cudaMalloc(pointer, bytes), operation);
}

bool malloc_managed(void** pointer, size_t bytes, const char* operation) {
    return report_cuda(cudaMallocManaged(pointer, bytes), operation);
}

bool copy_memory(void* destination, const void* source, size_t bytes,
                 cudaMemcpyKind kind, const char* operation) {
    return report_cuda(cudaMemcpy(destination, source, bytes, kind), operation);
}

}  // namespace mollm_cuda
