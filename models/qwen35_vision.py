"""Qwen3.5 vision encoder graph and weight conversion.

The Hugging Face patch embedding is a Conv3D whose kernel equals its stride:
([T, H, W] = [2, 16, 16]). The official image processor already emits one
flattened [C*T*H*W] row per non-overlapping patch, so the Conv3D is exported as
an exactly equivalent linear projection.

The first runtime implementation targets one image at a time. The patch count
is dynamic; Qwen's 2x2 spatial merger reduces it by an exact factor of four.
"""

from __future__ import annotations

import os

import numpy as np

from transpile import (
    DimExpr,
    GraphBuilder,
    Precision,
    SEQ,
    _write_weight_file,
)


def export_vision_weights(weights: dict[str, np.ndarray], weights_dir: str,
                          cfg: dict) -> None:
    """Export the dense Qwen3.5 vision tower in FP16 with FP32 biases/norms."""
    vc = cfg["vision_config"]
    os.makedirs(weights_dir, exist_ok=True)

    def get(name: str) -> np.ndarray:
        full = f"model.visual.{name}"
        if full not in weights:
            raise KeyError(f"Missing vision weight: {full}")
        return weights[full]

    def save(name: str, data: np.ndarray, fp32: bool = False) -> None:
        dtype = np.float32 if fp32 else np.float16
        _write_weight_file(
            os.path.join(weights_dir, f"vision_{name}.weights"),
            data.astype(dtype, copy=False))

    patch = get("patch_embed.proj.weight")
    expected = (
        vc["hidden_size"], vc["in_channels"], vc["temporal_patch_size"],
        vc["patch_size"], vc["patch_size"])
    if tuple(patch.shape) != expected:
        raise ValueError(
            f"unexpected vision patch weight {patch.shape}, expected {expected}")
    save("patch_embed_weight", patch.reshape(patch.shape[0], -1))
    save("patch_embed_bias", get("patch_embed.proj.bias"), fp32=True)
    # Kept as a graph constant so the runtime can interpolate it from grid_thw.
    save("pos_embed", get("pos_embed.weight"))

    for i in range(vc["depth"]):
        prefix = f"blocks.{i}"
        for norm in ("norm1", "norm2"):
            save(f"block_{i}_{norm}_weight",
                 get(f"{prefix}.{norm}.weight"), fp32=True)
            save(f"block_{i}_{norm}_bias",
                 get(f"{prefix}.{norm}.bias"), fp32=True)
        for linear in ("attn.qkv", "attn.proj",
                       "mlp.linear_fc1", "mlp.linear_fc2"):
            stem = linear.replace(".", "_")
            save(f"block_{i}_{stem}_weight",
                 get(f"{prefix}.{linear}.weight"))
            save(f"block_{i}_{stem}_bias",
                 get(f"{prefix}.{linear}.bias"), fp32=True)

    for name in ("norm.weight", "norm.bias"):
        save(f"merger_{name.replace('.', '_')}",
             get(f"merger.{name}"), fp32=True)
    for name in ("linear_fc1.weight", "linear_fc1.bias",
                 "linear_fc2.weight", "linear_fc2.bias"):
        save(f"merger_{name.replace('.', '_')}",
             get(f"merger.{name}"), fp32=name.endswith("bias"))


def build_vision_graph(weights_dir: str, cfg: dict,
                       build_patch_count: int = 256) -> GraphBuilder:
    """Build the single-image Qwen3.5 vision tower with dynamic patch count."""
    vc = cfg["vision_config"]
    hidden = vc["hidden_size"]
    heads = vc["num_heads"]
    head_dim = hidden // heads
    intermediate = vc["intermediate_size"]
    merge = vc["spatial_merge_size"]
    merge_unit = merge * merge
    patch_dim = (
        vc["in_channels"] * vc["temporal_patch_size"] *
        vc["patch_size"] * vc["patch_size"])
    out_hidden = vc["out_hidden_size"]
    seq = SEQ.bind(build_patch_count)
    dynamic_seq = (
        DimExpr.const(), DimExpr.seq(), DimExpr.const(), DimExpr.const())

    g = GraphBuilder()
    g.set_metadata("model_type", "qwen3_5_vision")
    g.set_metadata("vision_hidden_size", hidden)
    g.set_metadata("vision_num_heads", heads)
    g.set_metadata("vision_spatial_merge_size", merge)

    pixels = g.input(
        "pixel_values", (patch_dim, build_patch_count),
        dynamic=dynamic_seq)
    pos_embeds = g.input(
        "position_embeds", (hidden, build_patch_count),
        dynamic=dynamic_seq)
    cos = g.input(
        "vision_cos", (head_dim // 2, build_patch_count),
        dynamic=dynamic_seq)
    sin = g.input(
        "vision_sin", (head_dim // 2, build_patch_count),
        dynamic=dynamic_seq)

    def weight(name: str, shape: tuple, fp32: bool = False) -> int:
        return g.weight(
            os.path.join(weights_dir, f"vision_{name}.weights"), shape,
            Precision.FP32 if fp32 else Precision.FP16)

    def add_bias(x: int, name: str, size: int) -> int:
        return g.add(x, weight(name, (size,), fp32=True))

    w_patch = weight("patch_embed_weight", (hidden, patch_dim))
    x = add_bias(g.matmul(pixels, w_patch), "patch_embed_bias", hidden)

    # The table is intentionally present as an unconsumed constant. The
    # runtime reads it to construct the bilinearly interpolated input tensor.
    weight("pos_embed", (vc["num_position_embeddings"], hidden))
    x = g.add(x, pos_embeds)

    for i in range(vc["depth"]):
        residual = x
        norm1 = g.layer_norm(
            x,
            weight(f"block_{i}_norm1_weight", (hidden,), fp32=True),
            weight(f"block_{i}_norm1_bias", (hidden,), fp32=True),
            eps=1e-6)

        qkv_size = 3 * hidden
        qkv = g.matmul(
            norm1,
            weight(f"block_{i}_attn_qkv_weight",
                   (qkv_size, hidden)))
        qkv = add_bias(qkv, f"block_{i}_attn_qkv_bias", qkv_size)
        q, k, v = g.slice(qkv, [hidden, hidden, hidden], dim=0)
        q = g.reshape(q, (head_dim, heads, seq))
        k = g.reshape(k, (head_dim, heads, seq))
        v = g.reshape(v, (head_dim, heads, seq))
        q = g.permute(q, (0, 2, 1, 3))
        k = g.permute(k, (0, 2, 1, 3))
        v = g.permute(v, (0, 2, 1, 3))
        # The no-cache SDPA kernel consumes each V head as one contiguous
        # [sequence, head_dim] plane. Q/K become dense in the RoPE op below,
        # while V otherwise remains a strided token-major view.
        v = g.contiguous(v)
        q = g.rope(q, cos, sin, rope_dim=head_dim, interleave=False)
        k = g.rope(k, cos, sin, rope_dim=head_dim, interleave=False)
        attn = g.sdpa_no_cache(
            q, k, v, causal=False, scale=head_dim ** -0.5,
            num_heads=heads, num_kv_heads=heads,
            head_dim=head_dim, v_head_dim=head_dim)
        attn = g.permute(attn, (0, 2, 1, 3))
        attn = g.contiguous(attn)
        attn = g.reshape(attn, (hidden, seq))
        attn = g.matmul(
            attn,
            weight(f"block_{i}_attn_proj_weight", (hidden, hidden)))
        attn = add_bias(attn, f"block_{i}_attn_proj_bias", hidden)
        x = g.add(residual, attn)

        residual = x
        norm2 = g.layer_norm(
            x,
            weight(f"block_{i}_norm2_weight", (hidden,), fp32=True),
            weight(f"block_{i}_norm2_bias", (hidden,), fp32=True),
            eps=1e-6)
        # The reference is GELU(xW+b), so keep the activation explicit until
        # MATMUL supports fused bias+activation in that order.
        mlp_pre = g.matmul(
            norm2,
            weight(f"block_{i}_mlp_linear_fc1_weight",
                   (intermediate, hidden)))
        mlp_pre = add_bias(
            mlp_pre, f"block_{i}_mlp_linear_fc1_bias", intermediate)
        mlp = g.gelu(mlp_pre)
        mlp = g.matmul(
            mlp,
            weight(f"block_{i}_mlp_linear_fc2_weight",
                   (hidden, intermediate)))
        mlp = add_bias(mlp, f"block_{i}_mlp_linear_fc2_bias", hidden)
        x = g.add(residual, mlp)

    merged = g.layer_norm(
        x,
        weight("merger_norm_weight", (hidden,), fp32=True),
        weight("merger_norm_bias", (hidden,), fp32=True),
        eps=1e-6)
    merged = g.reshape(
        merged, (hidden * merge_unit, seq // merge_unit))
    merged_hidden = hidden * merge_unit
    merged = g.matmul(
        merged,
        weight("merger_linear_fc1_weight",
               (merged_hidden, merged_hidden)))
    merged = add_bias(
        merged, "merger_linear_fc1_bias", merged_hidden)
    merged = g.gelu(merged)
    merged = g.matmul(
        merged,
        weight("merger_linear_fc2_weight",
               (out_hidden, merged_hidden)))
    merged = add_bias(merged, "merger_linear_fc2_bias", out_hidden)
    return g
