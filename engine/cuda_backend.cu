#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "engine/cuda_backend.h"
#include "engine/cuda_internal.h"
#include "engine/engine.h"

#include "kernels/activations.h"
#include "kernels/quant_layouts.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct CudaBackend::Impl {
    enum class WeightLayout : uint8_t {
        Dense = 0,
        Q8RowMajor,
        Q4Bg32,
        Q4Bg128,
    };

    struct DeviceWeight {
        void* data = nullptr;
        float* scales = nullptr;
        cudaDataType type = CUDA_R_16F;
        int n = 0;
        int k = 0;
        int group_size = 0;
        int groups_per_row = 0;
        WeightLayout layout = WeightLayout::Dense;
    };

    struct BoundaryBuffer {
        void* data = nullptr;
        size_t capacity = 0;
    };

    bool ok = false;
    bool failed = false;
    cublasHandle_t cublas = nullptr;
    CPUBackend cpu;
    std::unordered_map<const void*, DeviceWeight> weights;
    std::unordered_map<const void*, const DeviceWeight*> weights_by_device;
    std::vector<void*> device_allocations;
    std::vector<void*> managed_allocations;
    std::unordered_map<void*, size_t> managed_sizes;
    std::multimap<size_t, void*> free_managed;
    std::unordered_map<std::string, BoundaryBuffer> boundary_buffers;
    std::unordered_map<uint32_t, uint64_t> native_ops;
    std::unordered_map<uint32_t, uint64_t> fallback_ops;
    OperatorFallbackPolicy operator_fallback =
        OperatorFallbackPolicy::ALLOW_REFERENCE;
    void* activation = nullptr;
    size_t activation_bytes = 0;
    void* activation_fp16 = nullptr;
    size_t activation_fp16_bytes = 0;
    void* quantized_weight_scratch = nullptr;
    size_t quantized_weight_scratch_bytes = 0;
    void* output = nullptr;
    size_t output_bytes = 0;
    void* attention_scores = nullptr;
    size_t attention_scores_bytes = 0;
    void* norm_scratch = nullptr;
    size_t norm_scratch_bytes = 0;

    ~Impl() {
        if (cublas)
            cublasDestroy(cublas);
        for (void* allocation : device_allocations)
            cudaFree(allocation);
        for (void* allocation : managed_allocations)
            cudaFree(allocation);
        for (auto& entry : boundary_buffers)
            if (entry.second.data)
                cudaFree(entry.second.data);
        if (activation)
            cudaFree(activation);
        if (activation_fp16)
            cudaFree(activation_fp16);
        if (quantized_weight_scratch)
            cudaFree(quantized_weight_scratch);
        if (output)
            cudaFree(output);
        if (attention_scores)
            cudaFree(attention_scores);
        if (norm_scratch)
            cudaFree(norm_scratch);
        if (std::getenv("MOLLM_CUDA_PROFILE")) {
            std::fprintf(stderr, "\nCudaBackend operator coverage:\n");
            for (const auto& entry : native_ops)
                std::fprintf(stderr, "  native   %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
            for (const auto& entry : fallback_ops)
                std::fprintf(stderr, "  fallback %-24s %llu\n",
                             op_type_name(static_cast<OpType>(entry.first)),
                             static_cast<unsigned long long>(entry.second));
        }
    }

    void* acquire_managed(size_t bytes) {
        auto found = free_managed.lower_bound(bytes);
        if (found != free_managed.end()) {
            void* pointer = found->second;
            free_managed.erase(found);
            return pointer;
        }
        void* pointer = nullptr;
        if (!mollm_cuda::malloc_managed(
                &pointer, bytes, "cudaMallocManaged output"))
            return nullptr;
        managed_allocations.push_back(pointer);
        managed_sizes.emplace(pointer, bytes);
        return pointer;
    }

    void release_managed(void* pointer) {
        const auto found = managed_sizes.find(pointer);
        if (found != managed_sizes.end())
            free_managed.emplace(found->second, pointer);
    }

    bool reserve(void*& pointer, size_t& capacity, size_t requested) {
        if (capacity >= requested)
            return true;
        if (pointer)
            cudaFree(pointer);
        pointer = nullptr;
        capacity = 0;
        if (!mollm_cuda::malloc_device(
                &pointer, requested, "cudaMalloc scratch"))
            return false;
        capacity = requested;
        return true;
    }

    const DeviceWeight* find_weight(const Tensor& tensor) const {
        if (!tensor.device_data)
            return nullptr;
        const auto found = weights_by_device.find(tensor.device_data);
        return found == weights_by_device.end() ? nullptr : found->second;
    }

    bool upload_weight(Tensor& tensor, const void* cache_key,
                       const void* source, size_t bytes, cudaDataType type,
                       int n, int k,
                       WeightLayout layout = WeightLayout::Dense) {
        if (!cache_key || !source || bytes == 0)
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end()) {
            void* device = nullptr;
            if (!mollm_cuda::malloc_device(
                    &device, bytes, "cudaMalloc weight") ||
                !mollm_cuda::copy_memory(
                    device, source, bytes, cudaMemcpyHostToDevice,
                    "cudaMemcpy weight")) {
                if (device)
                    cudaFree(device);
                return false;
            }
            device_allocations.push_back(device);
            found = weights.emplace(
                cache_key,
                DeviceWeight{device, nullptr, type, n, k, 0, 0, layout})
                        .first;
            weights_by_device.emplace(device, &found->second);
        }
        tensor.device_data = found->second.data;
        tensor.device_offset = 0;
        return true;
    }

    bool upload_quantized_weight(
        Tensor& tensor, const void* cache_key, const void* source,
        size_t bytes, const float* host_scales, size_t scale_count,
        int n, int k, int group_size, int groups_per_row) {
        if (!host_scales || scale_count == 0 || group_size <= 0 ||
            groups_per_row <= 0 ||
            !upload_weight(
                tensor, cache_key, source, bytes, CUDA_R_8I, n, k,
                WeightLayout::Q8RowMajor))
            return false;
        auto found = weights.find(cache_key);
        if (found == weights.end())
            return false;
        DeviceWeight& prepared = found->second;
        if (!prepared.scales) {
            float* device_scales = nullptr;
            const size_t scale_bytes = scale_count * sizeof(float);
            if (!mollm_cuda::malloc_device(
                    reinterpret_cast<void**>(&device_scales), scale_bytes,
                    "cudaMalloc weight scales") ||
                !mollm_cuda::copy_memory(
                    device_scales, host_scales, scale_bytes,
                    cudaMemcpyHostToDevice, "cudaMemcpy weight scales")) {
                if (device_scales)
                    cudaFree(device_scales);
                return false;
            }
            device_allocations.push_back(device_scales);
            prepared.scales = device_scales;
        }
        prepared.group_size = group_size;
        prepared.groups_per_row = groups_per_row;
        return true;
    }

    bool run_matmul_device(const float* device_a, int lda,
                           const Tensor& weight, float* device_c, int ldc,
                           int m, int n, int k,
                           Activation activation_kind, int act_begin,
                           int act_len) {
        const DeviceWeight* prepared = find_weight(weight);
        if (!prepared || prepared->n != n || prepared->k != k || !device_a ||
            !device_c || ldc != n)
            return false;

        const bool valid_q8 =
            prepared->layout == WeightLayout::Q8RowMajor &&
            prepared->scales && prepared->group_size > 0 &&
            prepared->groups_per_row ==
                (k + prepared->group_size - 1) / prepared->group_size;
        const bool valid_q4_g32 =
            prepared->layout == WeightLayout::Q4Bg32 && k % 32 == 0;
        const bool valid_q4_g128 =
            prepared->layout == WeightLayout::Q4Bg128 && k % 128 == 0;
        const bool valid_quantized =
            valid_q8 || valid_q4_g32 || valid_q4_g128;
        if (valid_quantized && m == 1) {
            const char* label = nullptr;
            if (valid_q8) {
                mollm_cuda::launch_q8_dense_gemv(
                    device_a, static_cast<const int8_t*>(prepared->data),
                    prepared->scales, prepared->group_size,
                    prepared->groups_per_row, device_c, n, k);
                label = "q8_dense_gemv_cuda";
            } else if (valid_q4_g32) {
                mollm_cuda::launch_q4_g32_dense_gemv(
                    device_a,
                    static_cast<const Q4B8G32Block*>(prepared->data),
                    device_c, n, k);
                label = "q4_g32_dense_gemv_cuda";
            } else {
                mollm_cuda::launch_q4_g128_dense_gemv(
                    device_a,
                    static_cast<const Q4B8G128Block*>(prepared->data),
                    device_c, n, k);
                label = "q4_g128_dense_gemv_cuda";
            }
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), label))
                return false;
        } else {
            const void* linear_weight = prepared->data;
            cudaDataType linear_weight_type = prepared->type;
            if (valid_quantized) {
                const size_t weight_elements = static_cast<size_t>(n) * k;
                if (!reserve(
                        quantized_weight_scratch,
                        quantized_weight_scratch_bytes,
                        weight_elements * sizeof(__half)))
                    return false;
                const char* label = nullptr;
                if (valid_q8) {
                    mollm_cuda::launch_dequantize_q8_dense_weight(
                        static_cast<const int8_t*>(prepared->data),
                        prepared->scales, prepared->group_size,
                        prepared->groups_per_row,
                        static_cast<__half*>(quantized_weight_scratch),
                        weight_elements, k);
                    label = "dequantize_q8_dense_weight_cuda";
                } else if (valid_q4_g32) {
                    const size_t packed_count =
                        static_cast<size_t>((n + 7) / 8) * (k / 32) * 8 * 16;
                    mollm_cuda::launch_dequantize_q4_g32_dense_weight(
                        static_cast<const Q4B8G32Block*>(prepared->data),
                        reinterpret_cast<__half2*>(quantized_weight_scratch),
                        packed_count, n, k / 32);
                    label = "dequantize_q4_g32_dense_weight_cuda";
                } else {
                    const size_t packed_count =
                        static_cast<size_t>((n + 7) / 8) * (k / 128) *
                        4 * 8 * 16;
                    mollm_cuda::launch_dequantize_q4_g128_dense_weight(
                        static_cast<const Q4B8G128Block*>(prepared->data),
                        reinterpret_cast<__half2*>(quantized_weight_scratch),
                        packed_count, n, k / 128);
                    label = "dequantize_q4_g128_dense_weight_cuda";
                }
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), label))
                    return false;
                linear_weight = quantized_weight_scratch;
                linear_weight_type = CUDA_R_16F;
            } else if (prepared->layout != WeightLayout::Dense) {
                return false;
            }

            const size_t a_elements = static_cast<size_t>(m) * lda;
            const void* gemm_activation = device_a;
            cudaDataType activation_type = CUDA_R_32F;
            if (linear_weight_type == CUDA_R_16F) {
                if (!reserve(activation_fp16, activation_fp16_bytes,
                             a_elements * sizeof(__half)))
                    return false;
                mollm_cuda::launch_fp32_to_fp16(
                    device_a, static_cast<__half*>(activation_fp16),
                    a_elements);
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "fp32_to_fp16"))
                    return false;
                gemm_activation = activation_fp16;
                activation_type = CUDA_R_16F;
            }

            if (!mollm_cuda::run_dense_matmul(
                    cublas, linear_weight, linear_weight_type,
                    gemm_activation, activation_type, device_c, m, n, k,
                    lda))
                return false;
        }

        if (activation_kind != Activation::NONE && act_len != 0) {
            const int begin = std::max(0, act_begin);
            const int end = act_len < 0 ? n : std::min(n, begin + act_len);
            mollm_cuda::launch_apply_activation(device_c, m, n,
                           static_cast<int>(activation_kind), begin, end);
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "apply_activation_cuda"))
                return false;
        }
        return true;
    }

    bool run_matmul(const float* host_a, int lda, const Tensor& weight,
                    float* host_c, int ldc, int m, int n, int k,
                    Activation activation_kind, int act_begin, int act_len) {
        const size_t a_bytes = static_cast<size_t>(m) * lda * sizeof(float);
        const size_t c_bytes = static_cast<size_t>(m) * n * sizeof(float);
        if (!host_a || !host_c ||
            !reserve(activation, activation_bytes, a_bytes) ||
            !reserve(output, output_bytes, c_bytes) ||
            !mollm_cuda::copy_memory(
                activation, host_a, a_bytes, cudaMemcpyHostToDevice,
                "cudaMemcpy activation") ||
            !run_matmul_device(
                static_cast<const float*>(activation), lda, weight,
                static_cast<float*>(output), ldc, m, n, k, activation_kind,
                act_begin, act_len) ||
            !mollm_cuda::copy_memory(
                host_c, output, c_bytes, cudaMemcpyDeviceToHost,
                "cudaMemcpy output"))
            return false;
        return true;
    }
};

namespace {

template <typename T>
T* device_pointer(const Tensor& tensor) {
    if (!tensor.device_data)
        return nullptr;
    return reinterpret_cast<T*>(
        static_cast<uint8_t*>(tensor.device_data) + tensor.device_offset);
}

template <typename T>
const T* device_pointer_const(const Tensor& tensor) {
    if (!tensor.device_data)
        return nullptr;
    return reinterpret_cast<const T*>(
        static_cast<const uint8_t*>(tensor.device_data) +
        tensor.device_offset);
}

bool fp32_contiguous(const Tensor& tensor) {
    return tensor.prec == Precision::FP32 && tensor.is_contiguous();
}

bool same_shape(const Tensor& lhs, const Tensor& rhs) {
    for (int dimension = 0; dimension < 4; ++dimension)
        if (lhs.shape[dimension] != rhs.shape[dimension])
            return false;
    return true;
}

}  // namespace

CudaBackend::CudaBackend() : impl_(std::make_unique<Impl>()) {
    int count = 0;
    if (!mollm_cuda::report_cuda(
            cudaGetDeviceCount(&count), "cudaGetDeviceCount") ||
        count <= 0)
        return;
    if (!mollm_cuda::report_cuda(cudaSetDevice(0), "cudaSetDevice") ||
        !mollm_cuda::report_cublas(
            cublasCreate(&impl_->cublas), "cublasCreate"))
        return;
    impl_->ok = true;
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
        std::fprintf(stderr, "CudaBackend: using %s (sm_%d%d)\n",
                     properties.name, properties.major, properties.minor);
}

CudaBackend::~CudaBackend() = default;

bool CudaBackend::available() const { return impl_ && impl_->ok; }

void CudaBackend::clear_dispatch_error() {
    impl_->failed = false;
    impl_->cpu.clear_dispatch_error();
}

bool CudaBackend::dispatch_failed() const {
    return impl_->failed || impl_->cpu.dispatch_failed();
}

bool CudaBackend::set_operator_fallback_policy(
    OperatorFallbackPolicy policy) {
    impl_->operator_fallback = policy;
    return true;
}

bool CudaBackend::register_weight_region(void*, size_t) { return true; }

void CudaBackend::wrap_weight(Tensor& tensor) {
    if (!available() || !tensor.data || tensor.shape[0] <= 0 ||
        tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);
    if (tensor.prec == Precision::FP16) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(__half),
                             CUDA_R_16F, n, k);
    } else if (tensor.prec == Precision::FP32) {
        impl_->upload_weight(tensor, tensor.data, tensor.data,
                             static_cast<size_t>(n) * k * sizeof(float),
                             CUDA_R_32F, n, k);
    } else if (tensor.prec == Precision::INT8 && tensor.scales &&
               tensor.group_size > 0) {
        const void* quantized =
            tensor.rowmajor_data ? tensor.rowmajor_data : tensor.data;
        const int group_size = static_cast<int>(tensor.group_size);
        const int groups_per_row = static_cast<int>(tensor.groups_per_row);
        if (quantized && groups_per_row ==
                (k + group_size - 1) / group_size) {
            impl_->upload_quantized_weight(
                tensor, quantized, quantized, static_cast<size_t>(n) * k,
                tensor.scales, static_cast<size_t>(n) * groups_per_row,
                n, k, group_size, groups_per_row);
        }
    }
}

void CudaBackend::wrap_weight_int4(Tensor& tensor,
                                   bool keep_native_experts) {
    if (!available() || keep_native_experts || tensor.prec != Precision::INT4 ||
        tensor.shape[0] <= 0 || tensor.shape[1] <= 0)
        return;
    const int n = static_cast<int>(tensor.shape[0]);
    const int k = static_cast<int>(tensor.shape[1]);

    if (tensor.is_q4_g32_packed && tensor.q4_g32_data && k % 32 == 0 &&
        tensor.group_size == 32 &&
        tensor.groups_per_row == static_cast<uint32_t>(k / 32)) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 32) *
            sizeof(Q4B8G32Block);
        impl_->upload_weight(
            tensor, tensor.q4_g32_data, tensor.q4_g32_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg32);
    } else if (tensor.is_q4_g128_packed && tensor.q4_g128_data &&
               k % 128 == 0 && tensor.group_size == 128 &&
               tensor.groups_per_row == static_cast<uint32_t>(k / 128)) {
        const size_t bytes = static_cast<size_t>((n + 7) / 8) * (k / 128) *
            sizeof(Q4B8G128Block);
        impl_->upload_weight(
            tensor, tensor.q4_g128_data, tensor.q4_g128_data, bytes,
            CUDA_R_8I, n, k, Impl::WeightLayout::Q4Bg128);
    }
}

void* CudaBackend::alloc_output(Tensor& output, size_t nbytes, BufferPool*) {
    if (!available() || nbytes == 0)
        return nullptr;
    void* pointer = impl_->acquire_managed(nbytes);
    if (!pointer) {
        impl_->failed = true;
        return nullptr;
    }
    output.data = pointer;
    output.device_data = pointer;
    output.device_offset = 0;
    output.mem_type = MemoryType::POOLED;
    output.owner_id = 0;
    output.storage_id = 0;
    return pointer;
}

void CudaBackend::free_output(Tensor& tensor, BufferPool*) {
    if (tensor.device_data)
        impl_->release_managed(tensor.device_data);
}

void CudaBackend::synchronize_for_host_read() {
    if (!mollm_cuda::report_cuda(
            cudaDeviceSynchronize(), "cudaDeviceSynchronize"))
        impl_->failed = true;
}

void CudaBackend::begin_graph() {}

void CudaBackend::end_graph() { synchronize_for_host_read(); }

void CudaBackend::alloc_persistent(
    Tensor& tensor, size_t nbytes, PersistentHostAccess host_access,
    size_t host_prefix_bytes) {
    (void)host_access;
    (void)host_prefix_bytes;
    void* storage = nullptr;
    if (!available() || nbytes == 0 ||
        !mollm_cuda::malloc_managed(
            &storage, nbytes, "cudaMallocManaged persistent")) {
        impl_->failed = true;
        return;
    }
    impl_->managed_allocations.push_back(storage);
    std::memset(storage, 0, nbytes);
    tensor.data = storage;
    tensor.device_data = storage;
    tensor.device_offset = 0;
    tensor.mem_type = MemoryType::EXTERNAL;
}

void CudaBackend::upload_input(Tensor& tensor, const std::string& key,
                               const void* host_source, size_t nbytes) {
    if (!available() || key.empty() || !host_source || nbytes == 0)
        return;
    auto& buffer = impl_->boundary_buffers[key];
    if (buffer.capacity < nbytes) {
        if (buffer.data)
            cudaFree(buffer.data);
        buffer.data = nullptr;
        buffer.capacity = 0;
        if (!mollm_cuda::malloc_device(
                &buffer.data, nbytes, "cudaMalloc input")) {
            impl_->failed = true;
            return;
        }
        buffer.capacity = nbytes;
    }
    if (!mollm_cuda::copy_memory(
            buffer.data, host_source, nbytes, cudaMemcpyHostToDevice,
            "cudaMemcpy input")) {
        impl_->failed = true;
        return;
    }
    tensor.device_data = buffer.data;
    tensor.device_offset = 0;
}

bool CudaBackend::supports_lm_head(const Tensor& weight) const {
    return impl_->find_weight(weight) != nullptr;
}

void CudaBackend::dispatch(const GraphNode& node,
                           const std::vector<const Tensor*>& inputs,
                           Tensor* output, ThreadPool* thread_pool) {
    auto record_native = [&]() {
        ++impl_->native_ops[static_cast<uint32_t>(node.op_type)];
    };
    if (node.op_type == OpType::INPUT ||
        node.op_type == OpType::CONSTANT) {
        record_native();
        return;
    }

    if (node.op_type == OpType::RESHAPE && !inputs.empty() && inputs[0] &&
        output && inputs[0]->is_contiguous()) {
        const int64_t shape[4] = {output->shape[0], output->shape[1],
                                  output->shape[2], output->shape[3]};
        *output = *inputs[0];
        for (int dimension = 0; dimension < 4; ++dimension)
            output->shape[dimension] = shape[dimension];
        output->compute_strides();
        record_native();
        return;
    }

    if (node.op_type == OpType::PERMUTE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int axis[4] = {
            graph_params::get_i32(node.params, 0, 0),
            graph_params::get_i32(node.params, 1, 1),
            graph_params::get_i32(node.params, 2, 2),
            graph_params::get_i32(node.params, 3, 3),
        };
        Tensor view = source;
        for (int dimension = 0; dimension < 4; ++dimension) {
            view.shape[axis[dimension]] = source.shape[dimension];
            view.stride[axis[dimension]] = source.stride[dimension];
        }
        *output = view;
        record_native();
        return;
    }

    if (node.op_type == OpType::SLICE && !inputs.empty() && inputs[0] &&
        output) {
        const Tensor& source = *inputs[0];
        const int dimension = graph_params::get_i32(node.params, 0, 0);
        const int offset = graph_params::get_i32(node.params, 1, 0);
        const int size = graph_params::get_i32(
            node.params, 2, static_cast<int>(source.shape[dimension]));
        *output = source;
        output->device_offset = source.device_offset +
            static_cast<size_t>(offset) * source.stride[dimension];
        output->shape[dimension] = size;
        record_native();
        return;
    }

    if ((node.op_type == OpType::CONTIGUOUS ||
         node.op_type == OpType::RESHAPE) &&
        !inputs.empty() && inputs[0] && output &&
        inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 && output->is_contiguous()) {
        const Tensor& source = *inputs[0];
        const float* input = device_pointer_const<float>(source);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (input && destination && source.nelements() == output->nelements()) {
            mollm_cuda::launch_contiguous(
                input, destination, count, source.shape[0], source.shape[1],
                source.shape[2], source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "contiguous_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ROTARY_EMBED && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& cosine = *inputs[1];
        const Tensor& sine = *inputs[2];
        const float* input = device_pointer_const<float>(source);
        const float* cos_data = device_pointer_const<float>(cosine);
        const float* sin_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int feature_dim = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(source.shape[1]);
        const int channels = static_cast<int>(source.shape[2] * source.shape[3]);
        const int rope_dim = graph_params::get_i32(node.params, 0, 64);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        if (input && cos_data && sin_data && destination &&
            rope_dim > 0 && rope_dim <= feature_dim && rope_dim % 2 == 0 &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2) {
            mollm_cuda::launch_rope(
                input, cos_data, sin_data, destination, feature_dim,
                sequence_length, channels, static_cast<int>(source.shape[2]),
                rope_dim, interleave,
                source.stride[0] / sizeof(float),
                source.stride[1] / sizeof(float),
                source.stride[2] / sizeof(float),
                source.stride[3] / sizeof(float),
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::SDPA ||
         node.op_type == OpType::SDPA_MLA) &&
        inputs.size() >= 3 && inputs[0] && inputs[1] && inputs[2] &&
        output && inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        inputs[2]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& query = *inputs[0];
        const Tensor& current_key = *inputs[1];
        const Tensor& current_value = *inputs[2];
        const Tensor* mask = inputs.size() > 3 && inputs[3] &&
                inputs[3]->data && inputs[3]->nelements() > 0
            ? inputs[3] : nullptr;
        const Tensor* key_cache = inputs.size() > 4 && inputs[4] &&
                inputs[4]->data
            ? inputs[4] : nullptr;
        const Tensor* value_cache = inputs.size() > 5 && inputs[5] &&
                inputs[5]->data
            ? inputs[5] : nullptr;
        const int cache_mode = graph_params::get_i32(node.params, 0, 2);
        const bool causal = graph_params::get_i32(node.params, 1, 1) != 0;
        const int num_heads = graph_params::get_i32(
            node.params, 2, static_cast<int>(query.shape[2]));
        const int num_kv_heads = graph_params::get_i32(
            node.params, 3, static_cast<int>(current_key.shape[2]));
        const int key_dim = graph_params::get_i32(
            node.params, 4, static_cast<int>(query.shape[0]));
        const int value_dim = graph_params::get_i32(
            node.params, 5, static_cast<int>(current_value.shape[0]));
        float scale = graph_params::get_f32(node.params, 0, 0.0f);
        if (scale == 0.0f)
            scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
        const int query_length = static_cast<int>(query.shape[1]);
        const int current_length = static_cast<int>(current_key.shape[1]);
        int past_length = 0;
        int key_capacity = current_length;
        bool cached = false;
        const float* key_data = device_pointer_const<float>(current_key);
        const float* value_data = device_pointer_const<float>(current_value);
        float* key_cache_data = nullptr;
        float* value_cache_data = nullptr;
        if (cache_mode == 2 && key_cache && value_cache &&
            key_cache->prec == Precision::FP32 &&
            value_cache->prec == Precision::FP32 &&
            key_cache->device_data && value_cache->device_data) {
            const auto* metadata = cache_meta(
                static_cast<const uint8_t*>(key_cache->data) +
                key_cache->device_offset);
            past_length = static_cast<int>(metadata->current_seq_len);
            key_capacity = static_cast<int>(metadata->max_seq_len);
            key_cache_data = reinterpret_cast<float*>(
                static_cast<uint8_t*>(key_cache->device_data) +
                key_cache->device_offset + CacheMetadata::SIZE);
            value_cache_data = reinterpret_cast<float*>(
                static_cast<uint8_t*>(value_cache->device_data) +
                value_cache->device_offset + CacheMetadata::SIZE);
            cached = true;
        }
        const int key_length = past_length + current_length;
        const float* query_data = device_pointer_const<float>(query);
        float* output_data = device_pointer<float>(*output);
        const float* mask_data = mask
            ? device_pointer_const<float>(*mask) : nullptr;
        const bool valid = query_data && key_data && value_data &&
            output_data && num_heads > 0 && num_kv_heads > 0 &&
            num_heads % num_kv_heads == 0 && query_length > 0 &&
            current_length > 0 && key_dim > 0 && value_dim > 0 &&
            query.shape[0] >= key_dim && current_key.shape[0] >= key_dim &&
            current_value.shape[0] >= value_dim &&
            output->shape[0] >= value_dim &&
            (cache_mode == 0 || cached) &&
            (!cached || (key_cache_data && value_cache_data &&
                         key_length <= key_capacity)) &&
            (!mask || mask_data);
        if (valid) {
            if (cached) {
                mollm_cuda::launch_append_kv(
                    key_data, value_data, key_cache_data, value_cache_data,
                    num_kv_heads, current_length, past_length, key_capacity,
                    key_dim, value_dim,
                    current_key.stride[1] / sizeof(float),
                    current_key.stride[2] / sizeof(float),
                    current_value.stride[1] / sizeof(float),
                    current_value.stride[2] / sizeof(float));
                if (!mollm_cuda::report_cuda(
                        cudaGetLastError(), "append_kv_cuda")) {
                    impl_->failed = true;
                    return;
                }
                key_data = key_cache_data;
                value_data = value_cache_data;
            }
            const size_t score_count = static_cast<size_t>(num_heads) *
                query_length * key_length;
            const size_t score_bytes = score_count * sizeof(float);
            if (!impl_->reserve(impl_->attention_scores,
                                impl_->attention_scores_bytes,
                                score_bytes)) {
                impl_->failed = true;
                return;
            }
            auto* scores = static_cast<float*>(impl_->attention_scores);
            mollm_cuda::launch_sdpa_scores(
                query_data, key_data, scores, mask_data, num_heads,
                num_kv_heads, query_length, key_length, past_length, key_dim,
                key_capacity, cached, causal, scale,
                query.stride[0] / sizeof(float),
                query.stride[1] / sizeof(float),
                query.stride[2] / sizeof(float),
                cached ? 1 : current_key.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(key_dim)
                       : current_key.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * key_dim
                       : current_key.stride[2] / sizeof(float),
                mask ? mask->stride[0] / sizeof(float) : 0,
                mask ? mask->stride[1] / sizeof(float) : 0);
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sdpa_scores_cuda")) {
                impl_->failed = true;
                return;
            }
            mollm_cuda::launch_sdpa_output(
                scores, value_data, output_data, num_heads, num_kv_heads,
                query_length, key_length, value_dim, key_capacity, cached,
                cached ? 1 : current_value.stride[0] / sizeof(float),
                cached ? static_cast<size_t>(value_dim)
                       : current_value.stride[1] / sizeof(float),
                cached ? static_cast<size_t>(key_capacity) * value_dim
                       : current_value.stride[2] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sdpa_output_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::MATMUL && inputs.size() >= 2 && inputs[0] &&
        inputs[1] && output && inputs[0]->prec == Precision::FP32 &&
        output->prec == Precision::FP32 &&
        impl_->find_weight(*inputs[1])) {
        const Tensor& a = *inputs[0];
        const Tensor& weight = *inputs[1];
        const int m = static_cast<int>(a.shape[1]);
        const int k = static_cast<int>(a.shape[0]);
        const int n = static_cast<int>(weight.shape[0]);
        const int lda = static_cast<int>(a.stride[1] / sizeof(float));
        const int ldc = static_cast<int>(output->stride[1] / sizeof(float));
        const Activation activation_kind = static_cast<Activation>(
            graph_params::get_i32(node.params, 0, 0));
        const float* device_a = device_pointer_const<float>(a);
        float* device_c = device_pointer<float>(*output);
        const bool ok = device_a && device_c
            ? impl_->run_matmul_device(
                  device_a, lda, weight, device_c, ldc, m, n, k,
                  activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1))
            : impl_->run_matmul(
                  a.ptr<float>(), lda, weight, output->ptr<float>(), ldc,
                  m, n, k, activation_kind,
                  graph_params::get_i32(node.params, 1, 0),
                  graph_params::get_i32(node.params, 2, -1));
        if (ok) {
            record_native();
            return;
        }
        impl_->failed = true;
        return;
    }

    if (node.op_type == OpType::RMS_NORM_ROPE && inputs.size() >= 4 &&
        inputs[0] && inputs[1] && inputs[2] && inputs[3] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        inputs[2]->prec == Precision::FP32 &&
        inputs[3]->prec == Precision::FP32 &&
        output->prec == Precision::FP32) {
        const Tensor& source = *inputs[0];
        const Tensor& weight = *inputs[1];
        const Tensor& cosine = *inputs[2];
        const Tensor& sine = *inputs[3];
        const float* source_data = device_pointer_const<float>(source);
        const float* weight_data = device_pointer_const<float>(weight);
        const float* cosine_data = device_pointer_const<float>(cosine);
        const float* sine_data = device_pointer_const<float>(sine);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(source.shape[0]);
        const int sequence_length = static_cast<int>(output->shape[1]);
        const int channels =
            static_cast<int>(output->shape[2] * output->shape[3]);
        const int rows = width > 0
            ? static_cast<int>(source.nelements() / width) : 0;
        const int rope_dim = graph_params::get_i32(node.params, 0, width);
        const bool interleave =
            graph_params::get_i32(node.params, 1, 1) != 0;
        const size_t count = static_cast<size_t>(source.nelements());
        const size_t scratch_bytes = count * sizeof(float);
        if (source_data && weight_data && cosine_data && sine_data &&
            destination && width > 0 && rows > 0 && rope_dim > 0 &&
            rope_dim <= width && rope_dim % 2 == 0 &&
            rows == sequence_length * channels &&
            cosine.shape[0] >= rope_dim / 2 &&
            sine.shape[0] >= rope_dim / 2 &&
            impl_->reserve(impl_->norm_scratch,
                           impl_->norm_scratch_bytes, scratch_bytes)) {
            auto* normalized = static_cast<float*>(impl_->norm_scratch);
            mollm_cuda::launch_rms_norm(
                source_data, weight_data, normalized, width, rows,
                graph_params::get_f32(node.params, 0, 1e-6f));
            mollm_cuda::launch_rope(
                normalized, cosine_data, sine_data, destination, width,
                sequence_length, channels, static_cast<int>(output->shape[2]),
                rope_dim, interleave, 1, static_cast<size_t>(width),
                static_cast<size_t>(width) * sequence_length,
                static_cast<size_t>(width) * sequence_length *
                    output->shape[2],
                cosine.stride[0] / sizeof(float),
                cosine.stride[1] / sizeof(float),
                sine.stride[0] / sizeof(float),
                sine.stride[1] / sizeof(float),
                output->stride[0] / sizeof(float),
                output->stride[1] / sizeof(float),
                output->stride[2] / sizeof(float),
                output->stride[3] / sizeof(float));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "rms_norm_rope_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::ADD_RMS_NORM && inputs.size() >= 3 &&
        inputs[0] && inputs[1] && inputs[2] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 &&
        fp32_contiguous(*inputs[2]) && output->prec == Precision::FP32 &&
        inputs[0]->stride[0] == sizeof(float) &&
        inputs[1]->stride[0] == sizeof(float) &&
        output->stride[0] == sizeof(float) &&
        inputs[0]->shape[2] == 1 && inputs[0]->shape[3] == 1 &&
        inputs[1]->shape[2] == 1 && inputs[1]->shape[3] == 1 &&
        output->shape[2] == 1 && output->shape[3] == 1 &&
        inputs[0]->nelements() == inputs[1]->nelements() &&
        inputs[0]->nelements() == output->nelements()) {
        float* residual = device_pointer<float>(*inputs[0]);
        const float* update = device_pointer_const<float>(*inputs[1]);
        const float* weight = device_pointer_const<float>(*inputs[2]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = static_cast<int>(inputs[0]->shape[1]);
        if (residual && update && weight && destination && width > 0 &&
            rows > 0) {
            mollm_cuda::launch_add_rms_norm(
                residual, update, weight, destination, width, rows,
                inputs[0]->stride[1] / sizeof(float),
                inputs[1]->stride[1] / sizeof(float),
                output->stride[1] / sizeof(float),
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "add_rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::RMS_NORM && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output && fp32_contiguous(*inputs[0]) &&
        fp32_contiguous(*inputs[1]) && fp32_contiguous(*output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        const float* weight = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const int width = static_cast<int>(inputs[0]->shape[0]);
        const int rows = width > 0
            ? static_cast<int>(inputs[0]->nelements() / width) : 0;
        if (source && weight && destination && rows > 0) {
            mollm_cuda::launch_rms_norm(
                source, weight, destination, width, rows,
                graph_params::get_f32(node.params, 0, 1e-6f));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "rms_norm_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if ((node.op_type == OpType::ADD || node.op_type == OpType::MUL) &&
        inputs.size() >= 2 && inputs[0] && inputs[1] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*inputs[1]) &&
        fp32_contiguous(*output) &&
        inputs[0]->nelements() == inputs[1]->nelements()) {
        const float* lhs = device_pointer_const<float>(*inputs[0]);
        const float* rhs = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (lhs && rhs && destination) {
            mollm_cuda::launch_binary(
                lhs, rhs, destination, count,
                node.op_type == OpType::MUL);
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "binary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SIGMOID_MUL && inputs.size() >= 2 &&
        inputs[0] && inputs[1] && output &&
        inputs[0]->prec == Precision::FP32 &&
        inputs[1]->prec == Precision::FP32 && fp32_contiguous(*output) &&
        same_shape(*inputs[0], *inputs[1]) &&
        inputs[0]->nelements() == output->nelements()) {
        const float* value = device_pointer_const<float>(*inputs[0]);
        const float* gate = device_pointer_const<float>(*inputs[1]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (value && gate && destination) {
            if (inputs[0]->is_contiguous() && inputs[1]->is_contiguous()) {
                mollm_cuda::launch_sigmoid_mul(value, gate, destination, count);
            } else {
                mollm_cuda::launch_sigmoid_mul_strided(
                    value, gate, destination, count, inputs[0]->shape[0],
                    inputs[0]->shape[1], inputs[0]->shape[2],
                    inputs[0]->stride[0] / sizeof(float),
                    inputs[0]->stride[1] / sizeof(float),
                    inputs[0]->stride[2] / sizeof(float),
                    inputs[0]->stride[3] / sizeof(float),
                    inputs[1]->stride[0] / sizeof(float),
                    inputs[1]->stride[1] / sizeof(float),
                    inputs[1]->stride[2] / sizeof(float),
                    inputs[1]->stride[3] / sizeof(float));
            }
            if (!mollm_cuda::report_cuda(
                    cudaGetLastError(), "sigmoid_mul_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    int unary_operation = -1;
    switch (node.op_type) {
    case OpType::SILU: unary_operation = 0; break;
    case OpType::GELU: unary_operation = 1; break;
    case OpType::TANH: unary_operation = 2; break;
    case OpType::SIGMOID:
    case OpType::SIGMOID_EXACT: unary_operation = 3; break;
    case OpType::EXP:
    case OpType::EXP_EXACT: unary_operation = 4; break;
    case OpType::SOFTPLUS: unary_operation = 5; break;
    default: break;
    }
    if (unary_operation >= 0 && !inputs.empty() && inputs[0] && output &&
        fp32_contiguous(*inputs[0]) && fp32_contiguous(*output)) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            mollm_cuda::launch_unary(
                source, destination, count, unary_operation);
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "unary_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (node.op_type == OpType::SWIGLU && !inputs.empty() && inputs[0] &&
        output && fp32_contiguous(*inputs[0]) && fp32_contiguous(*output) &&
        inputs[0]->shape[0] == output->shape[0] * 2) {
        const float* source = device_pointer_const<float>(*inputs[0]);
        float* destination = device_pointer<float>(*output);
        const size_t count = static_cast<size_t>(output->nelements());
        if (source && destination) {
            mollm_cuda::launch_swiglu(
                source, destination, count,
                static_cast<size_t>(output->shape[0]));
            if (!mollm_cuda::report_cuda(cudaGetLastError(), "swiglu_cuda")) {
                impl_->failed = true;
                return;
            }
            record_native();
            return;
        }
    }

    if (impl_->operator_fallback ==
        OperatorFallbackPolicy::REQUIRE_NATIVE) {
        std::fprintf(
            stderr,
            "CudaBackend: native-only mode rejected %s operator fallback\n",
            op_type_name(node.op_type));
        impl_->failed = true;
        return;
    }

    synchronize_for_host_read();
    if (impl_->failed)
        return;
    std::vector<Tensor> host_tensors;
    std::vector<const Tensor*> host_inputs;
    host_tensors.reserve(inputs.size());
    host_inputs.reserve(inputs.size());
    for (const Tensor* input : inputs) {
        if (!input) {
            host_inputs.push_back(nullptr);
            continue;
        }
        host_tensors.push_back(*input);
        Tensor& host = host_tensors.back();
        if (host.data && host.device_offset)
            host.data = static_cast<uint8_t*>(host.data) +
                host.device_offset;
        host.device_data = nullptr;
        host.device_offset = 0;
        host_inputs.push_back(&host);
    }
    impl_->cpu.clear_dispatch_error();
    impl_->cpu.dispatch(node, host_inputs, output, thread_pool);
    if (impl_->cpu.dispatch_failed()) {
        impl_->failed = true;
        return;
    }
    ++impl_->fallback_ops[static_cast<uint32_t>(node.op_type)];
}

void CudaBackend::lm_head_gemv(const float* activation_host,
                               const Tensor& weight, float* output_host,
                               int n, int k, int activation) {
    if (!impl_->run_matmul(
            activation_host, k, weight, output_host, n, 1, n, k,
            static_cast<Activation>(activation), 0, -1))
        impl_->failed = true;
}

void CudaBackend::lm_head_gemv_device_and_end_graph(
    const Tensor& activation, size_t activation_element_offset,
    const Tensor& weight, float* output_host, int n, int k,
    int activation_kind) {
    const float* activation_base = device_pointer_const<float>(activation);
    const float* source = activation_base
        ? activation_base + activation_element_offset : nullptr;
    const size_t output_size = static_cast<size_t>(n) * sizeof(float);
    if (!source ||
        !impl_->reserve(impl_->output, impl_->output_bytes, output_size) ||
        !impl_->run_matmul_device(
            source, k, weight, static_cast<float*>(impl_->output), n,
            1, n, k, static_cast<Activation>(activation_kind), 0, -1) ||
        !mollm_cuda::copy_memory(
            output_host, impl_->output, output_size, cudaMemcpyDeviceToHost,
            "cudaMemcpy lm_head output")) {
        impl_->failed = true;
        return;
    }
    ++impl_->native_ops[static_cast<uint32_t>(OpType::MATMUL)];
}
