"""Convert a native DeepSeek-V4-Flash checkpoint to a mollm package."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from safetensors_stream import (
    SafeTensorIndex,
    aggregate_mxfp4_experts,
    dense_streamed_weight,
    fp32_streamed_weight,
    integer_streamed_weight,
)
from transpile import (
    DimExpr,
    GraphBuilder,
    Precision,
    save_package,
)


def _ref(name: str) -> str:
    return f"./{name}.weights"


def _add_engine_boundary_weights(graphs: tuple[GraphBuilder, ...],
                                 streamed: dict, index: SafeTensorIndex,
                                 cfg: dict):
    embed_ref = _ref("embed_tokens")
    head_ref = _ref("lm_head")
    streamed[embed_ref] = dense_streamed_weight(index, "embed.weight")
    streamed[head_ref] = dense_streamed_weight(index, "head.weight")
    shape = (int(cfg["vocab_size"]), int(cfg["hidden_size"]))
    for graph in graphs:
        graph.weight(embed_ref, shape, Precision.FP16)
        graph.weight(head_ref, shape, Precision.FP16)


def _add_layer_moe_weights(
        graph: GraphBuilder, cfg: dict, streamed: dict,
        index: SafeTensorIndex, layer: int) -> tuple[int, ...]:
    hidden = int(cfg["hidden_size"])
    experts = int(cfg["n_routed_experts"])
    intermediate = int(cfg["moe_intermediate_size"])
    top_k = int(cfg["num_experts_per_tok"])

    router_ref = _ref(f"layer_{layer}_router")
    gate_up_ref = _ref(f"layer_{layer}_experts_gate_up")
    down_ref = _ref(f"layer_{layer}_experts_down")
    shared_gate_ref = _ref(f"layer_{layer}_shared_gate")
    shared_up_ref = _ref(f"layer_{layer}_shared_up")
    shared_down_ref = _ref(f"layer_{layer}_shared_down")

    streamed[router_ref] = fp32_streamed_weight(
        index, f"layers.{layer}.ffn.gate.weight")
    streamed[gate_up_ref] = aggregate_mxfp4_experts(
        index,
        (f"layers.{layer}.ffn.experts.{expert}.{projection}.weight"
         for expert in range(experts)
         for projection in ("w1", "w3")),
        interleave_expert_count=experts)
    streamed[down_ref] = aggregate_mxfp4_experts(
        index,
        (f"layers.{layer}.ffn.experts.{expert}.w2.weight"
         for expert in range(experts)),
        interleave_expert_count=experts)
    streamed[shared_gate_ref] = dense_streamed_weight(
        index, f"layers.{layer}.ffn.shared_experts.w1.weight")
    streamed[shared_up_ref] = dense_streamed_weight(
        index, f"layers.{layer}.ffn.shared_experts.w3.weight")
    streamed[shared_down_ref] = dense_streamed_weight(
        index, f"layers.{layer}.ffn.shared_experts.w2.weight")

    router = graph.weight(
        router_ref, (experts, hidden), Precision.FP32)
    gate_up = graph.weight(
        gate_up_ref, (experts * 2 * intermediate, hidden),
        Precision.MXFP4)
    down = graph.weight(
        down_ref, (experts * hidden, intermediate), Precision.MXFP4)
    shared_gate = graph.weight(
        shared_gate_ref, (intermediate, hidden), Precision.FP8_E4M3)
    shared_up = graph.weight(
        shared_up_ref, (intermediate, hidden), Precision.FP8_E4M3)
    shared_down = graph.weight(
        shared_down_ref, (hidden, intermediate), Precision.FP8_E4M3)
    router_bias = None
    hash_table = None
    if layer < int(cfg["num_hash_layers"]):
        hash_ref = _ref(f"layer_{layer}_tid2eid")
        streamed[hash_ref] = integer_streamed_weight(
            index, f"layers.{layer}.ffn.gate.tid2eid",
            logical_shape=(top_k, int(cfg["vocab_size"])))
        hash_table = graph.weight(
            hash_ref, (top_k, int(cfg["vocab_size"])), Precision.INT32)
    else:
        bias_ref = _ref(f"layer_{layer}_router_bias")
        streamed[bias_ref] = fp32_streamed_weight(
            index, f"layers.{layer}.ffn.gate.bias")
        router_bias = graph.weight(
            bias_ref, (experts,), Precision.FP32)
    return (
        router, gate_up, down, shared_gate, shared_up, shared_down,
        router_bias, hash_table)


def _attention_weight(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        layer: int, suffix: str, shape: tuple[int, ...],
        precision: Precision, fp32: bool = False) -> int:
    label = f"layer_{layer}_attn_{suffix.replace('.', '_')}"
    reference = _ref(label)
    source_name = f"layers.{layer}.attn.{suffix}"
    streamed[reference] = (
        fp32_streamed_weight(index, source_name, logical_shape=shape)
        if fp32 else dense_streamed_weight(index, source_name))
    return graph.weight(reference, shape, precision)


def _add_attention_branch(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        cfg: dict, layer: int, hidden: int, position: int, n_tokens: int,
        n_ctx: int) -> int:
    """Append one checkpoint-backed DeepSeek-V4 attention branch."""
    hidden_size = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    head_dim = int(cfg["head_dim"])
    q_rank = int(cfg["q_lora_rank"])
    o_rank = int(cfg["o_lora_rank"])
    groups = int(cfg["o_groups"])
    window = int(cfg["sliding_window"])
    ratio = int(cfg["compress_ratios"][layer])
    rope_dim = int(cfg["qk_rope_head_dim"])
    top_k = int(cfg["index_topk"])
    index_heads = int(cfg["index_n_heads"])
    index_dim = int(cfg["index_head_dim"])
    eps = float(cfg["rms_norm_eps"])
    rope_scaling = cfg["rope_scaling"]
    original_context = (
        int(rope_scaling["original_max_position_embeddings"])
        if ratio else 0)
    rope_theta = float(
        cfg["compress_rope_theta"] if ratio else cfg["rope_theta"])
    rope_factor = float(rope_scaling["factor"] if ratio else 1.0)
    beta_fast = float(rope_scaling["beta_fast"])
    beta_slow = float(rope_scaling["beta_slow"])

    def weight(suffix: str, shape: tuple[int, ...],
               precision: Precision, fp32: bool = False) -> int:
        return _attention_weight(
            graph, streamed, index, layer, suffix, shape, precision, fp32)

    wq_a = weight(
        "wq_a.weight", (q_rank, hidden_size), Precision.FP8_E4M3)
    q_norm = weight(
        "q_norm.weight", (q_rank,), Precision.FP32, fp32=True)
    wq_b = weight(
        "wq_b.weight", (heads * head_dim, q_rank),
        Precision.FP8_E4M3)
    wkv = weight(
        "wkv.weight", (head_dim, hidden_size), Precision.FP8_E4M3)
    kv_norm = weight(
        "kv_norm.weight", (head_dim,), Precision.FP32, fp32=True)
    sink = weight(
        "attn_sink", (heads,), Precision.FP32, fp32=True)
    wo_a = weight(
        "wo_a.weight", (groups * o_rank, heads * head_dim // groups),
        Precision.FP8_E4M3)
    wo_b = weight(
        "wo_b.weight", (hidden_size, groups * o_rank),
        Precision.FP8_E4M3)

    q_lora = graph.rms_norm(
        graph.matmul(hidden, wq_a), q_norm, eps)
    query = graph.matmul(q_lora, wq_b)
    current_kv = graph.rms_norm(
        graph.matmul(hidden, wkv), kv_norm, eps)

    state_base = layer * 16
    window_cache = graph.input(
        f"aux_state{state_base}", (head_dim, window),
        prec=Precision.FP32)
    compressed_cache = None
    compressed_indices = None
    dependencies: list[int] = []
    if ratio:
        overlap = ratio == 4
        coff = 2 if overlap else 1
        projected = coff * head_dim
        state_rows = coff * ratio
        capacity = n_ctx // ratio
        compressor_wkv = weight(
            "compressor.wkv.weight", (projected, hidden_size),
            Precision.FP32, fp32=True)
        compressor_wgate = weight(
            "compressor.wgate.weight", (projected, hidden_size),
            Precision.FP32, fp32=True)
        compressor_ape = weight(
            "compressor.ape", (projected, ratio),
            Precision.FP32, fp32=True)
        compressor_norm = weight(
            "compressor.norm.weight", (head_dim,),
            Precision.FP32, fp32=True)
        compressor_kv_state = graph.input(
            f"aux_state{state_base + 1}", (projected, state_rows),
            prec=Precision.FP32)
        compressor_score_state = graph.input(
            f"aux_state{state_base + 2}", (projected, state_rows),
            prec=Precision.FP32)
        compressed_cache = graph.input(
            f"aux_state{state_base + 3}", (head_dim, capacity),
            prec=Precision.FP32)
        compressor_dependency = graph.dsv4_compressor(
            hidden, compressor_wkv, compressor_wgate, compressor_ape,
            compressor_norm, compressor_kv_state,
            compressor_score_state, compressed_cache, position, n_tokens,
            hidden_size, head_dim, ratio, overlap, False, rope_dim,
            original_context, eps, rope_theta, rope_factor,
            beta_fast, beta_slow)
        dependencies.append(compressor_dependency)

        if ratio == 4:
            index_projected = 2 * index_dim
            index_state_rows = 2 * ratio
            index_wq_b = weight(
                "indexer.wq_b.weight", (index_heads * index_dim, q_rank),
                Precision.FP8_E4M3)
            index_weights = weight(
                "indexer.weights_proj.weight", (index_heads, hidden_size),
                Precision.FP16)
            index_wkv = weight(
                "indexer.compressor.wkv.weight",
                (index_projected, hidden_size), Precision.FP32, fp32=True)
            index_wgate = weight(
                "indexer.compressor.wgate.weight",
                (index_projected, hidden_size), Precision.FP32, fp32=True)
            index_ape = weight(
                "indexer.compressor.ape", (index_projected, ratio),
                Precision.FP32, fp32=True)
            index_norm = weight(
                "indexer.compressor.norm.weight", (index_dim,),
                Precision.FP32, fp32=True)
            index_kv_state = graph.input(
                f"aux_state{state_base + 4}",
                (index_projected, index_state_rows),
                prec=Precision.FP32)
            index_score_state = graph.input(
                f"aux_state{state_base + 5}",
                (index_projected, index_state_rows),
                prec=Precision.FP32)
            index_cache = graph.input(
                f"aux_state{state_base + 6}", (index_dim, capacity),
                prec=Precision.FP32)
            compressed_indices = graph.dsv4_indexer(
                hidden, q_lora, index_wq_b, index_weights,
                index_wkv, index_wgate, index_ape, index_norm,
                index_kv_state, index_score_state, index_cache,
                position, n_tokens, hidden_size, q_rank, index_heads,
                index_dim, top_k, ratio, True, True, rope_dim,
                original_context, eps, rope_theta, rope_factor,
                beta_fast, beta_slow)

    attended = graph.dsv4_sparse_attention(
        query, current_kv, sink, window_cache, position, n_tokens,
        heads, head_dim, window, ratio, top_k, rope_dim,
        original_context, head_dim ** -0.5, eps, rope_theta,
        rope_factor, beta_fast, beta_slow,
        compressed_cache=compressed_cache,
        compressed_indices=compressed_indices,
        dependencies=dependencies)
    projected = graph.dsv4_grouped_linear(attended, wo_a, groups)
    return graph.matmul(projected, wo_b)


def _fp32_graph_weight(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        reference: str, source: str, shape: tuple[int, ...]) -> int:
    path = _ref(reference)
    streamed[path] = fp32_streamed_weight(
        index, source, logical_shape=shape)
    return graph.weight(path, shape, Precision.FP32)


def _add_moe_branch(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        cfg: dict, layer: int, hidden: int, token_ids: int) -> int:
    (router, gate_up, down, shared_gate, shared_up, shared_down,
     router_bias, hash_table) = _add_layer_moe_weights(
        graph, cfg, streamed, index, layer)
    return graph.moe(
        hidden, router, gate_up, down,
        shared_gate, shared_up, shared_down,
        shared_expert_gate=None,
        hidden_size=int(cfg["hidden_size"]),
        num_experts=int(cfg["n_routed_experts"]),
        top_k=int(cfg["num_experts_per_tok"]),
        intermediate_size=int(cfg["moe_intermediate_size"]),
        shared_intermediate_size=int(cfg["moe_intermediate_size"]),
        router_bias=router_bias,
        router_score_func=2,  # sqrt(softplus)
        norm_topk_prob=bool(cfg["norm_topk_prob"]),
        has_shared_expert=True,
        shared_expert_has_gate=False,
        routed_scaling_factor=float(cfg["routed_scaling_factor"]),
        swiglu_limit=float(cfg["swiglu_limit"]),
        hash_token_ids=token_ids if hash_table is not None else None,
        hash_table=hash_table)


def _add_hc_branch(
        graph: GraphBuilder, streamed: dict, index: SafeTensorIndex,
        cfg: dict, layer: int, kind: str, residual: int,
        branch_builder) -> int:
    hidden = int(cfg["hidden_size"])
    hc_mult = int(cfg["hc_mult"])
    wide = hidden * hc_mult
    mix = (2 + hc_mult) * hc_mult
    prefix = f"layers.{layer}.hc_{kind}"
    fn = _fp32_graph_weight(
        graph, streamed, index, f"layer_{layer}_hc_{kind}_fn",
        f"{prefix}_fn", (mix, wide))
    scale = _fp32_graph_weight(
        graph, streamed, index, f"layer_{layer}_hc_{kind}_scale",
        f"{prefix}_scale", (3,))
    base = _fp32_graph_weight(
        graph, streamed, index, f"layer_{layer}_hc_{kind}_base",
        f"{prefix}_base", (mix,))
    packed = graph.hc_pre(
        residual, fn, scale, base, hidden, hc_mult,
        int(cfg["hc_sinkhorn_iters"]), float(cfg["rms_norm_eps"]),
        float(cfg["hc_eps"]))
    reduced = graph.slice_range(packed, 0, hidden, dim=0)
    branch = branch_builder(reduced)
    return graph.hc_post(branch, residual, packed, hidden, hc_mult)


def build_full_graph(
        model_dir: str | Path, cfg: dict, seq_len: int,
        n_ctx: int) -> tuple[GraphBuilder, dict]:
    index = SafeTensorIndex(Path(model_dir))
    streamed: dict = {}
    graph = GraphBuilder()
    hidden_size = int(cfg["hidden_size"])
    hc_mult = int(cfg["hc_mult"])
    if seq_len == 1:
        hidden = graph.input(
            "hidden", (hidden_size, 1), prec=Precision.FP32)
        token_ids = graph.input(
            "token_ids", (1,), prec=Precision.INT32)
    else:
        hidden = graph.input(
            "hidden", (hidden_size, seq_len), prec=Precision.FP32,
            dynamic=(DimExpr.const(), DimExpr.seq()))
        token_ids = graph.input(
            "token_ids", (seq_len,), prec=Precision.INT32,
            dynamic=(DimExpr.seq(),))
    position = graph.input("position", (1,), prec=Precision.INT32)
    n_tokens = graph.input("n_tokens", (1,), prec=Precision.INT32)
    state = graph.tile(hidden, (hc_mult, 1))
    eps = float(cfg["rms_norm_eps"])

    for layer in range(int(cfg["num_hidden_layers"])):
        attn_norm = _fp32_graph_weight(
            graph, streamed, index, f"layer_{layer}_attn_norm",
            f"layers.{layer}.attn_norm.weight", (hidden_size,))

        def attention_branch(reduced: int, *, current_layer=layer,
                             norm=attn_norm) -> int:
            normalized = graph.rms_norm(reduced, norm, eps)
            return _add_attention_branch(
                graph, streamed, index, cfg, current_layer, normalized,
                position, n_tokens, n_ctx)

        state = _add_hc_branch(
            graph, streamed, index, cfg, layer, "attn", state,
            attention_branch)

        ffn_norm = _fp32_graph_weight(
            graph, streamed, index, f"layer_{layer}_ffn_norm",
            f"layers.{layer}.ffn_norm.weight", (hidden_size,))

        def ffn_branch(reduced: int, *, current_layer=layer,
                       norm=ffn_norm) -> int:
            normalized = graph.rms_norm(reduced, norm, eps)
            return _add_moe_branch(
                graph, streamed, index, cfg, current_layer,
                normalized, token_ids)

        state = _add_hc_branch(
            graph, streamed, index, cfg, layer, "ffn", state,
            ffn_branch)

    wide = hidden_size * hc_mult
    head_fn = _fp32_graph_weight(
        graph, streamed, index, "hc_head_fn", "hc_head_fn",
        (hc_mult, wide))
    head_scale = _fp32_graph_weight(
        graph, streamed, index, "hc_head_scale", "hc_head_scale", (1,))
    head_base = _fp32_graph_weight(
        graph, streamed, index, "hc_head_base", "hc_head_base", (hc_mult,))
    collapsed = graph.hc_head(
        state, head_fn, head_scale, head_base, hidden_size, hc_mult,
        eps, float(cfg["hc_eps"]))
    final_norm = _fp32_graph_weight(
        graph, streamed, index, "final_norm", "norm.weight",
        (hidden_size,))
    output = graph.rms_norm(collapsed, final_norm, eps)
    return graph, streamed


def convert_full(
        model_dir: str, output_path: str,
        prefill_seq_len: int = 32, n_ctx: int = 16384):
    model_dir_path = Path(model_dir)
    cfg = json.loads((model_dir_path / "config.json").read_text())
    if cfg.get("model_type") != "deepseek_v4":
        raise ValueError("checkpoint is not DeepSeek-V4")
    prefill, prefill_weights = build_full_graph(
        model_dir_path, cfg, prefill_seq_len, n_ctx)
    decode, decode_weights = build_full_graph(
        model_dir_path, cfg, 1, n_ctx)
    streamed = dict(prefill_weights)
    streamed.update(decode_weights)
    index = SafeTensorIndex(model_dir_path)
    _add_engine_boundary_weights(
        (prefill, decode), streamed, index, cfg)

    layers = []
    for layer in range(int(cfg["num_hidden_layers"])):
        layers.append({
            "layer": layer,
            "num_experts": int(cfg["n_routed_experts"]),
            "gate_up": {
                "weight": _ref(f"layer_{layer}_experts_gate_up"),
                "rows_per_expert":
                    2 * int(cfg["moe_intermediate_size"]),
                "cols": int(cfg["hidden_size"]),
            },
            "down": {
                "weight": _ref(f"layer_{layer}_experts_down"),
                "rows_per_expert": int(cfg["hidden_size"]),
                "cols": int(cfg["moe_intermediate_size"]),
            },
        })
    metadata = {
        "model_name": "DeepSeek-V4-Flash",
        "architecture": "deepseek-v4",
        "num_layers": int(cfg["num_hidden_layers"]),
        "hidden_size": int(cfg["hidden_size"]),
        "num_heads": int(cfg["num_attention_heads"]),
        "num_kv_heads": 1,
        "head_dim": int(cfg["head_dim"]),
        "rope_dim": int(cfg["qk_rope_head_dim"]),
        "rope_theta": float(cfg["rope_theta"]),
        "prefill_seq_len": prefill_seq_len,
        "n_ctx": n_ctx,
        "vocab_size": int(cfg["vocab_size"]),
        "num_experts": int(cfg["n_routed_experts"]),
        "quantization": "native-fp8-mxfp4",
        "moe_expert_storage": {
            "version": 1,
            "layout": "expert_interleaved_v1",
            "num_experts": int(cfg["n_routed_experts"]),
            "layers": layers,
        },
    }
    with tempfile.TemporaryDirectory(prefix="mollm_dsv4_full_") as empty_dir:
        save_package(
            output_path, prefill, decode, empty_dir, metadata,
            tokenizer_path=str(model_dir_path / "tokenizer.json"),
            streamed_weights=streamed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("output")
    parser.add_argument("--prefill-seq-len", type=int, default=32)
    parser.add_argument("--n-ctx", type=int, default=16384)
    args = parser.parse_args()
    convert_full(
        args.model_dir, args.output,
        args.prefill_seq_len, args.n_ctx)
