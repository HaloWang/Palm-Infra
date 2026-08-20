# Running Large MoE models with SSD offload

mollm runs Qwen3.5-122B-A10B, DeepSeek-V4-Flash, and Hy3-295B-A21B on a 48GB
Apple Silicon Mac without keeping the complete model resident. Always-used
dense weights stay locked in RAM, while asynchronous `pread` workers fetch only
the routed MoE expert pairs from the package. A bounded RAM cache, configured
with `--ssd-cache-mb`, retains recently
used expert pairs.

## Cache policy

The default uses one shared cache instead of fixed per-layer quotas. Eviction
adapts to cache pressure: when a layer's fair share cannot hold its current
route, entries from layers already executed in this pass are recycled first;
when the route fits, Least-Stale ordering instead preserves cross-token hits.
A next-layer router prediction submits low-priority reads before the following
MoE layer needs them.

On macOS, the expert cache is pinned by default. Without pinning, the VM
compressor may compress cold anonymous expert buffers and later decompress them
inside matmul. This can make a larger cache slower despite a higher hit rate.
Use `--no-lock-expert-cache` when system-memory headroom matters more than stable
decode throughput.

Historical cache-policy isolation run: real chat prompt, four CPU threads, 16 prompt tokens, 128 generated tokens,
`warmup=1`, and five independent process runs:

| 16 GiB cache policy | Decode | Expert-cache hit rate | Avg. SSD reads / generated token |
|---|---:|---:|---:|
| Legacy equal per-layer cache, no prediction | 10.89 t/s | 74.5% | 0.54 GB/token |
| **Shared cache + cross-layer prefetch** | **12.70 t/s** | **89.3%** | 0.59 GB/token |

The default row locks both dense weights and the expert cache. The prediction
uses somewhat more SSD bandwidth but hides enough latency to improve
interactive decode. With a longer 256-token context, the strict `pp256 + tg64`,
`warmup=3` five-process median refreshed on 2026-07-27 is 51.06 pp / 13.11 tg.

## Cache-size sweep

This sweep was rerun on 2026-07-29 using the prompt “给我讲一个故事”, greedy
decoding, 16 prompt tokens, 256 generated tokens, `warmup=0`, and three
independent processes per cache size.

| Expert RAM cache | Decode | Peak RSS | Expert-cache hit rate | Avg. SSD reads / generated token |
|---:|---:|---:|---:|---:|
| **1 GiB** | 12.38 t/s | **5.90 GiB** | 47.9% | 1.72 GB/token |
| **10 GiB** | 16.19 t/s | 14.64 GiB | 83.5% | 0.75 GB/token |
| **16 GiB** | **16.53 t/s** | 20.60 GiB | **88.6%** | **0.51 GB/token** |

Ten GiB is within 2.1% of the 16 GiB throughput while using about 6 GiB less
peak RSS. The 1 GiB configuration demonstrates that the 122B package can run
at about 6 GiB peak RSS, at the cost of substantially more SSD I/O.

The tables above use the original combined-run counter: logical routed-expert
bytes successfully loaded by demand or prefetch across both prompt prefill and
decode, divided by generated tokens. It excludes dense-weight and CPU-sidecar
loading and is not a direct measurement of filesystem or physical-device
traffic. In particular, it must not be combined with decode-only throughput to
estimate a bandwidth roofline.

Current `mollm_bench` output takes a statistics snapshot after prefill and
resets counters before timed decode. It reports separate `moe_ssd_prefill_*`
and `moe_ssd_decode_*` keys. Within each phase:

- `logical_load` is the logical size of cache entries whose load completed;
- `demand_origin_load` and `prefetch_origin_load` split entries by the request
  that originally created their residency;
- `useful_prefetch` counts prefetched entries when first consumed;
- `unused_prefetch_evicted` counts prefetched entries evicted before use;
- `expert_bytes_acquired` counts routed-expert bytes handed to compute,
  including cache hits;
- `wait_ms` is foreground time blocked in expert acquisition.
- `slot_waits` and `slot_wait_ms` isolate waits caused by every eviction
  candidate still loading; zero means foreground time was spent waiting for
  the requested expert itself rather than for cache capacity.

Loads and usefulness can occur in different phases. For example, a prefetch
completed near the end of prefill appears in prefill load bytes, while its
first use can appear in decode useful-prefetch bytes. All byte counters remain
logical application-level quantities rather than physical NAND traffic.

Cache capacity must leave room for dense weights, KV cache, runtime buffers, and
other applications.

## Trace SSD overlap

Both `mollm_chat` and `mollm_bench` accept `--trace <path.json>` and write a
Chrome Trace / Perfetto timeline. It includes prefill/decode, per-layer routing
and expert compute, cache requests and acquisition, merged worker `pread`
operations, and flow arrows from queued reads to their workers.

```bash
./build/mollm_bench \
  --package /path/to/qwen35_122b_a10b_w4g128_ssd.mollm \
  --ssd-cache-mb 16384 --ssd-io-workers 8 --threads 4 \
  --prompt "Give me a short story" --max-new-tokens 128 --warmup 1 \
  --trace /tmp/mollm_122b_trace.json
```

Open the resulting JSON in [Perfetto](https://ui.perfetto.dev/).

![Chrome Trace / Perfetto view of decode, MoE execution, and SSD I/O workers](../assets/ssd_io_trace.png)

Future work includes cache-aware prefetch admission and issuing routed-expert
reads during attention, before the MoE layer begins.
