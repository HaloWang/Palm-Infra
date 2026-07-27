# palm-infra

Palm Team 的 AI Infra 项目，目前包含 `mollm`。完整英文说明见 [README.md](README.md)。

## mollm

mobile-oriented LLM inference engine.
```
                 _ _
 _ __ ___   ___ | | |_ __ ___
| '_ ` _ \ / _ \| | | '_ ` _ \
| | | | | | (_) | | | | | | | |
|_| |_| |_|\___/|_|_|_| |_| |_|
```

`mollm` 是面向 ARM CPU 的轻量 C++ LLM 推理引擎，并提供实验性的 Apple Metal 支持。它将已支持的 Hugging Face 模型目录转换成单个 `.mollm` 文件，其中包含计算图、权重、tokenizer 与对话模板，并可直接运行。

项目当前聚焦 Apple Silicon 和其他现代 ARM 处理器上的高性能本地推理：FP16 使用 NEON FP16FML kernel；CPU 量化模型使用针对 ARM dot-product 指令优化的 weight-only int8/int4 kernel。

## 现在，48GB Mac 也能运行 122B 模型

`mollm` 可以在 48GB Apple Silicon Mac 上运行 W4 量化的
Qwen3.5-122B-A10B：稠密权重保留在 RAM 中，仅从 SSD 读取被路由到的 MoE
expert。在当前 256-token cache sweep 中，通过有界的 16 GiB 共享 expert
cache 和跨层预取，decode 达到 16.22 t/s。

cache 容量可配置，并不要求把模型完整常驻内存。在下面的真实 prompt sweep
中，1 GiB expert cache 配置的峰值 RSS 仅 5.90 GiB；增大 cache 可以用更多内存
换取更少的 SSD 读取和更高吞吐。

| Expert RAM cache | Decode | Peak RSS | Cache 命中率 | SSD 读取量 |
|---:|---:|---:|---:|---:|
| **1 GiB** | 11.35 t/s | **5.91 GiB** | 0.0% | 555.0 GB |
| **10 GiB** | 16.15 t/s | 14.64 GiB | 83.9% | 191.4 GB |
| **16 GiB** | **16.22 t/s** | 20.59 GiB | **89.8%** | **118.3 GB** |

该 sweep 使用 16-token 真实 prompt、生成 256 tokens、greedy decoding、
`warmup=0`，每档 cache 取三个独立进程的中位数，并于 2026-07-26 重新运行。
10 GiB cache 的 decode 吞吐与 16 GiB 相差不到 0.5%，同时峰值 RSS 少约
6 GiB；1 GiB 则展示了低内存运行能力。

cache 策略、内存/吞吐 sweep、I/O 行为和 Perfetto trace 详见
[122B MoE SSD offload](docs/ssd-offload.md)。

## 已支持的模型

| 模型系列 | 状态 |
|---|---|
| Qwen3 dense text models | FP16、W8、W4 |
| Qwen3-30B-A3B MoE | 仅文本 W4 路径 |
| Qwen3.6-35B-A3B MoE | 仅文本 W4 路径 |
| Qwen3.5-122B-A10B MoE | CPU W4，支持 SSD expert offload |
| Qwen3.5-0.8B / Qwen3.5-4B | FP16、W8、W4、混合 W4 |
| Youtu-LLM-2B | FP16、W8、W4、混合 W4 |
| RWKV7 | FP16、W8、混合 W4；循环式 CPU prefill/decode |

当前测试最充分的运行路径是 `w4g128`：它占用内存最少，且具有 mollm 中最快的
decode 速度。所有基于 `config.json` 的 converter 也支持 `w4g32` 和
`w4mixg32`。更小的 group 能改善 W4 质量，代价是更多 scale 和潜在的吞吐下降；
mixed 模式沿用各模型 `w4mixg128` 的 policy，决定哪些敏感 tensor 保留为 W8。

## 性能

### CPU

除非另有说明，Apple M5 Pro 数据使用 4 CPU 线程、`pp256 + tg64`、
`warmup=3` 和独立进程中位数。

当前 W4 中位数在 Youtu-LLM-2B 和 Qwen3.5-4B 上分别达到 140.77 和
69.78 decode tokens/s，是对应 llama.cpp Q4_0 CPU 结果的 1.47 倍和
1.73 倍。W4A8 decode 将激活与权重统一为 128-value 分组，先在整数寄存器
中累计四个 dot-product block，再统一应用 scale。图中的所有 dense 与
recurrent W4 行均使用当前模型包与二进制重新测试。

![CPU 吞吐量对比](assets/performance_cpu.svg)

### Metal（实验性）

Apple M5 Pro Metal 数据采用相同的 `pp256 + tg64`、`warmup=3` 和独立
进程中位数协议。Decode 从已有 256-token 上下文开始，两个运行时都将
模型权重保留在 Metal 上。

![Metal 吞吐量对比](assets/performance_metal.svg)

[测试协议、完整 CPU/Metal 性能表、长上下文 scaling 与正确性门禁](docs/performance.md)

## 快速开始

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j

# W4 转换需要此工具。
cmake --build build_i8mm --target mollm-quantize

# 转换 Hugging Face 模型目录。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# 从单个包文件启动对话。
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

交互命令：

```text
/reset   清空对话上下文
/quit    退出
```

## 构建

依赖：

- macOS/Apple Silicon 或 ARM Linux
- CMake 与 Ninja 或 Make
- Python 3
- 转换所需的 Python 包，主要是 `numpy` 与 `safetensors`

推荐构建方式：

```bash
cmake -G Ninja -B build_i8mm -DCMAKE_BUILD_TYPE=Release
cmake --build build_i8mm -j
```

若编译器和 CPU 支持 ARM i8mm，构建系统会自动启用更快的 int8 GEMM 路径。普通 `build/` 目录也可以使用；将示例中的 `build_i8mm` 替换为对应目录即可。

## 转换模型

转换器会从 `config.json` 自动识别模型类型。

```bash
# 默认 FP16 包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_fp16.mollm

# W8 int8 基线。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w8pc.mollm w8pc

# W4 性能包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g128.mollm w4g128

# 混合 W4 质量包。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg128.mollm w4mixg128

# 更小 group 的纯 W4，以及沿用同一模型 mixed policy 的版本。
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4g32.mollm w4g32
python3 models/converter.py /path/to/Qwen3.5-4B qwen35_4b_w4mixg32.mollm w4mixg32

# RWKV7
python3 models/rwkv7.py /path/to/rwkv7-g1h-1.5b.pth rwkv7_1.5b_w4mixg32.mollm --tokenizer /path/to/tokenizer.txt --quant w4mixg32
```

MoE 示例：

```bash
python3 models/converter.py \
    /path/to/Qwen3-30B-A3B \
    qwen3_30b_a3b_w4g128.mollm \
    w4g128
```

| `model_type` | 支持的模型 |
|---|---|
| `qwen3` | Qwen3 dense text models |
| `qwen3_moe` | Qwen3 MoE text models |
| `qwen3_5` | Qwen3.5 dense text models |
| `qwen3_5_moe` | Qwen3.5/3.6 MoE text models |
| `youtu` | Youtu-LLM MLA models |

| 模式 | 适用场景 |
|---|---|
| `fp16` | 最简单的基线，且内存充足。 |
| `w8pc` | 需要 weight-only int8 量化，允许轻微质量偏移。 |
| `w4g128` | 需要最小包大小和最快 decode；通常是性能首选。 |
| `w4mixg128` | 纯 W4 质量不足，可使用更多内存保留部分 W8 tensor。 |
| `w4g32` | 使用更小的 32-value group 改善 W4 质量，并接受更多 scale 和潜在吞吐损失。 |
| `w4mixg32` | 在 W4G32 基础上，沿用该模型 `w4mixg128` 的 W8 tensor promotion policy。 |

W4 转换需要 C++ 构建的 `mollm-quantize` 工具；FP16 和 W8 不需要。prefill
图内部以 256 token 为分块大小，但 CPU runtime 使用 dynamic prefill；除非
显式指定 `--static-padded`，短 prompt 不会 padding 到 256。

## 运行对话

```bash
./build_i8mm/mollm_chat --package qwen35_4b_w4g128.mollm --threads 4
```

一次性、确定性的 smoke test：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --prompt "请只输出一句话，不要解释：杭州有什么特点？" \
    --max-new-tokens 64 \
    --threads 4 \
    --temperature 0
```

采样生成：

```bash
./build_i8mm/mollm_chat \
    --package qwen35_4b_w4g128.mollm \
    --temperature 0.8 --top-p 0.95 --top-k 40 --min-p 0.05 \
    --repeat-penalty 1.05 --repeat-last-n 128 --seed 42
```

同时支持 OpenAI 风格的 `--presence-penalty` 与
`--frequency-penalty`。参数范围和默认值可通过 `mollm_chat --help` 查看。

默认情况下，`mollm_chat` 以 resident 模式加载包内权重。若需 mmap A/B 测试，传入 `--mmap`；默认 mmap 页面 warmup 已启用，可搭配 `--no-load-warmup` 关闭。

## 基准测试

```bash
./build_i8mm/mollm_bench \
    --package qwen35_4b_w4g128.mollm \
    --prompt-tokens 256 \
    --max-new-tokens 64 \
    --warmup 3 \
    --threads 4
```

## 本地 HTTP 服务

```bash
./build_i8mm/mollm_server \
    --package qwen35_4b_w4g128.mollm \
    --host 127.0.0.1 --port 8080 --threads 4
```

初始 server 提供 `GET /v1/models` 和 OpenAI 兼容的 `POST /v1/chat/completions`（含 SSE streaming）。它在串行请求间保留一个精确 token-prefix KV cache。每个请求均可指定采样参数；默认仍为确定性的 `temperature=0`。详见 [SERVER.md](SERVER.md) 的字段、示例与限制。

## 项目结构

```text
mollm/
├── kernels/    matmul、attention、MoE、norm、rope 的 ARM kernel
├── graph/      计算图格式、执行器、mmap 包加载、BufferPool
├── engine/     LLMEngine、tokenizer、对话/生成生命周期
├── models/     Python 转换器与计算图构建器
├── examples/   mollm_chat、mollm_server、mollm_bench、mollm_ppl
└── tests/      单元、压力与端到端测试
```

## 路线图

- 优化 prefill 性能，特别是 W8/W4 稠密模型的 prompt 处理。
- 提升实验性 Metal 性能，重点是量化 prefill、MoE prefill 与 CPU/GPU 同步开销。
- 基于当前单用户 REPL cache，为 serving 工作负载实现完整的 prefix cache。
- 扩展 accelerator 覆盖范围，同时保持 CPU runtime 作为可移植基线。
- 增加更多模型系列、视觉模型支持以及 SSD offload。

## 许可证

Copyright 2026 Tencent。根据 Apache License 2.0 发布；详见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。捆绑依赖的声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 致谢

- [Cider](https://github.com/Mininglamp-AI/cider)：其在 Apple Silicon 上使用 Metal 4 INT8 TensorOps 实现 W8A8/W4A8 推理的工作，为 mollm 实验性的量化 Metal 路径提供了启发。
- Fang 等人的 [Fate](https://arxiv.org/abs/2502.12224)：其跨层 gate 预测思路为 mollm 实验性的 expert 预取路径提供了启发。
