// Microbenchmark for the decode-time Metal SSD-MoE routed-expert kernel.
//
// The buffers use the same Shared storage and native BG128 package layout as
// the full-Metal SSD path. This isolates selected GEMV compute from model
// loading, routing, I/O scheduling, and the rest of the graph.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "kernels/metal/metal_common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef MOLLM_METALLIB_PATH
#define MOLLM_METALLIB_PATH ""
#endif

namespace {

constexpr size_t kBg128Bytes = 544;

struct Case {
    const char* name;
    int n;
    int k;
    int activation_rows;
    int activation_repeat;
};

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "bench_metal_moe: %s\n", message);
    std::exit(1);
}

id<MTLBuffer> make_buffer(id<MTLDevice> device, size_t bytes,
                          const char* label,
                          MTLResourceOptions options =
                              MTLResourceStorageModeShared) {
    id<MTLBuffer> buffer =
        [device newBufferWithLength:bytes options:options];
    if (!buffer) fail("Metal buffer allocation failed");
    buffer.label = [NSString stringWithUTF8String:label];
    return buffer;
}

void initialize_weights(id<MTLBuffer> buffer, size_t expert_bytes,
                        int selections) {
    auto* data = static_cast<uint8_t*>(buffer.contents);
    for (int selection = 0; selection < selections; ++selection) {
        auto* expert = data + static_cast<size_t>(selection) * expert_bytes;
        const size_t blocks = expert_bytes / kBg128Bytes;
        for (size_t block_index = 0; block_index < blocks; ++block_index) {
            auto* block = expert + block_index * kBg128Bytes;
            auto* scales = reinterpret_cast<float*>(block);
            for (int channel = 0; channel < 8; ++channel)
                scales[channel] =
                    0.0025f * static_cast<float>(1 + channel % 3);
            for (size_t byte = 32; byte < kBg128Bytes; ++byte)
                block[byte] =
                    static_cast<uint8_t>((byte * 17 + block_index * 13 +
                                          selection * 7) &
                                         0xff);
        }
    }
}

double run_case(id<MTLDevice> device, id<MTLCommandQueue> queue,
                id<MTLComputePipelineState> pipeline, const Case& c,
                int selections, int warmup, int iterations,
                bool private_weights, bool tensor_kernel, int layers) {
    const int groups_per_row = c.k / 128;
    const size_t expert_bytes =
        static_cast<size_t>(c.n / 8) * groups_per_row * kBg128Bytes;
    const size_t dispatch_weight_bytes = expert_bytes * selections;
    const size_t weight_bytes = dispatch_weight_bytes * layers;
    const size_t activation_bytes =
        static_cast<size_t>(c.activation_rows) * c.k;
    const size_t output_elements =
        static_cast<size_t>(selections) * c.n;

    id<MTLBuffer> activation =
        make_buffer(device, activation_bytes, "benchmark activations");
    id<MTLBuffer> weight_staging =
        make_buffer(device, weight_bytes, "benchmark expert weight staging");
    id<MTLBuffer> weights = private_weights
        ? make_buffer(device, weight_bytes, "benchmark expert weights",
                      MTLResourceStorageModePrivate)
        : weight_staging;
    id<MTLBuffer> output =
        make_buffer(device, output_elements * sizeof(float),
                    "benchmark expert output");
    id<MTLBuffer> activation_scales =
        make_buffer(device,
                    static_cast<size_t>(c.activation_rows) * sizeof(float),
                    "benchmark activation scales");
    id<MTLBuffer> weight_offsets =
        make_buffer(device,
                    static_cast<size_t>(layers) * selections *
                        sizeof(uint64_t),
                    "benchmark weight offsets");
    id<MTLBuffer> selection_indices =
        make_buffer(device,
                    static_cast<size_t>(layers) * selections *
                        sizeof(uint32_t),
                    "benchmark selection indices");

    auto* activation_data = static_cast<int8_t*>(activation.contents);
    for (size_t i = 0; i < activation_bytes; ++i)
        activation_data[i] = static_cast<int8_t>((i * 29 + 11) % 255 - 127);
    initialize_weights(weight_staging, expert_bytes, selections * layers);
    auto* scale_data = static_cast<float*>(activation_scales.contents);
    for (int row = 0; row < c.activation_rows; ++row)
        scale_data[row] = 0.003f * static_cast<float>(row + 1);
    auto* offsets_data = static_cast<uint64_t*>(weight_offsets.contents);
    auto* indices_data = static_cast<uint32_t*>(selection_indices.contents);
    for (int layer = 0; layer < layers; ++layer) {
        for (int selection = 0; selection < selections; ++selection) {
            const size_t index =
                static_cast<size_t>(layer) * selections + selection;
            offsets_data[index] =
                static_cast<uint64_t>(index) * expert_bytes;
            indices_data[index] = tensor_kernel
                ? static_cast<uint32_t>(index)
                : static_cast<uint32_t>(selection);
        }
    }
    if (private_weights) {
        id<MTLCommandBuffer> upload = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [upload blitCommandEncoder];
        [blit copyFromBuffer:weight_staging
                sourceOffset:0
                    toBuffer:weights
           destinationOffset:0
                        size:weight_bytes];
        [blit endEncoding];
        [upload commit];
        [upload waitUntilCompleted];
        if (upload.status != MTLCommandBufferStatusCompleted)
            fail("Metal private-weight upload failed");
    }

    SelectedW4A8Params params{};
    params.selections = selections;
    params.N = c.n;
    params.K = c.k;
    params.c_offset = 0;
    params.c_row_stride = c.n;
    params.group_size = 128;
    params.groups_per_row = groups_per_row;
    params.rows_per_expert = c.n;
    params.activation_rows = c.activation_rows;
    params.activation_repeat = c.activation_repeat;

    auto encode = ^(id<MTLCommandBuffer> command_buffer, int repeats) {
        id<MTLComputeCommandEncoder> encoder =
            [command_buffer computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:activation offset:0 atIndex:0];
        [encoder setBuffer:weights offset:0 atIndex:1];
        [encoder setBuffer:output offset:0 atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        [encoder setBuffer:activation_scales offset:0 atIndex:4];
        if (tensor_kernel) {
            [encoder setBuffer:weights offset:0 atIndex:5];
            [encoder setThreadgroupMemoryLength:
                         2 * 64 * 64 * sizeof(int32_t)
                                            atIndex:0];
        }
        for (int repeat = 0; repeat < repeats; ++repeat) {
            const NSUInteger metadata_offset =
                static_cast<NSUInteger>(repeat % layers) * selections;
            if (tensor_kernel) {
                [encoder setBuffer:selection_indices
                            offset:metadata_offset * sizeof(uint32_t)
                           atIndex:6];
            } else {
                [encoder setBuffer:weight_offsets
                            offset:metadata_offset * sizeof(uint64_t)
                           atIndex:6];
                [encoder setBuffer:selection_indices
                            offset:metadata_offset * sizeof(uint32_t)
                           atIndex:7];
            }
            if (tensor_kernel) {
                [encoder dispatchThreadgroups:
                             MTLSizeMake(1, (c.n + 63) / 64, selections)
                    threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [encoder dispatchThreadgroups:
                             MTLSizeMake((c.n + 31) / 32, 1, selections)
                    threadsPerThreadgroup:MTLSizeMake(32, 4, 1)];
            }
        }
        [encoder endEncoding];
    };

    if (warmup > 0) {
        id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
        encode(command_buffer, warmup);
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status != MTLCommandBufferStatusCompleted)
            fail("Metal warmup command failed");
    }

    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    command_buffer.label =
        [NSString stringWithFormat:@"bench_metal_moe %s", c.name];
    encode(command_buffer, iterations);
    const auto wall_start = std::chrono::steady_clock::now();
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    const auto wall_end = std::chrono::steady_clock::now();
    if (command_buffer.status != MTLCommandBufferStatusCompleted)
        fail("Metal benchmark command failed");

    const double gpu_ms =
        (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();
    const double gpu_us = gpu_ms * 1000.0 / iterations;
    const double wall_us = wall_ms * 1000.0 / iterations;
    const double bandwidth =
        static_cast<double>(dispatch_weight_bytes) /
        (gpu_us * 1e-6) / 1e9;
    const volatile float checksum =
        static_cast<const float*>(output.contents)[output_elements - 1];

    std::printf(
        "%-8s %-7s %-6s K=%-5d N=%-5d experts=%d layers=%d "
        "weights=%5.1f MiB "
        "gpu=%8.1f us wall=%8.1f us effective=%6.1f GB/s checksum=%g\n",
        c.name, private_weights ? "private" : "shared",
        tensor_kernel ? "tensor" : "native",
        c.k, c.n, selections, layers,
        static_cast<double>(dispatch_weight_bytes) / (1024.0 * 1024.0),
        gpu_us,
        wall_us, bandwidth, static_cast<double>(checksum));
    return gpu_us;
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        int iterations = 20;
        int warmup = 5;
        if (argc > 1) iterations = std::max(1, std::atoi(argv[1]));
        if (argc > 2) warmup = std::max(0, std::atoi(argv[2]));
        const bool private_weights =
            argc > 3 && std::string(argv[3]) == "private";
        const bool tensor_kernel =
            argc > 4 && std::string(argv[4]) == "tensor";
        const int layers = tensor_kernel
            ? 1
            : (argc > 5 ? std::max(1, std::atoi(argv[5])) : 8);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) fail("no Metal device");
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) fail("could not create Metal command queue");

        NSString* library_path =
            [NSString stringWithUTF8String:MOLLM_METALLIB_PATH];
        NSError* error = nil;
        id<MTLLibrary> library =
            [device newLibraryWithURL:[NSURL fileURLWithPath:library_path]
                               error:&error];
        if (!library) {
            std::fprintf(stderr, "metallib load failed: %s\n",
                         error.localizedDescription.UTF8String);
            return 1;
        }
        NSString* function_name =
            tensor_kernel
                ? @"gemm_selected_w4a8_i8a_i4b_f32c"
                : @"gemv_selected_slots_bg128_i8a_i4b_f32c";
        id<MTLFunction> function =
            [library newFunctionWithName:function_name];
        if (!function) fail("selected BG128 kernel is missing");
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:function
                                                   error:&error];
        if (!pipeline) {
            std::fprintf(stderr, "pipeline creation failed: %s\n",
                         error.localizedDescription.UTF8String);
            return 1;
        }

        std::printf("device=%s iterations=%d warmup=%d weights=%s kernel=%s\n",
                    device.name.UTF8String, iterations, warmup,
                    private_weights ? "private" : "shared",
                    tensor_kernel ? "tensor" : "native");
        const Case cases[] = {
            {"gate_up", 2048, 3072, 1, 8},
            {"down", 3072, 1024, 8, 1},
        };
        for (const Case& c : cases)
            run_case(device, queue, pipeline, c, 8, warmup, iterations,
                     private_weights, tensor_kernel, layers);
    }
    return 0;
}
