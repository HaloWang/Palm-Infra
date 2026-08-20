"""Generic converter for Qwen3-style decoder MoE text models.

This converter is intentionally named by architecture, not by any downstream
fork. It supports:
  - Qwen3 decoder full attention with GQA
  - dense MLP layers before first_k_dense_replace
  - routed MoE layers with per-expert HF weights
  - sigmoid router scoring with optional correction bias
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from qwen35 import _canonical_quant, _quant_spec
from transpile import (
    GraphBuilder,
    Precision,
    _write_weight_file,
    quantize_weight_w8_group,
    save_package,
    write_quantized_weight_file_cpp,
)


ROUTER_SCORE_SOFTMAX = 0
ROUTER_SCORE_SIGMOID = 1


def _num_experts(cfg: dict) -> int:
    if "n_routed_experts" in cfg:
        return int(cfg["n_routed_experts"])
    return int(cfg["num_experts"])


def _norm_suffixes(cfg: dict) -> tuple[str, str]:
    kind = cfg.get("_attn_norm_kind", "q_layernorm")
    if kind == "q_norm":
        return "q_norm", "k_norm"
    return "q_layernorm", "k_layernorm"


def _has_router_bias(cfg: dict) -> bool:
    return bool(cfg.get("_has_router_correction_bias", False))


def _num_shared_experts(cfg: dict) -> int:
    if "n_shared_experts" in cfg:
        return int(cfg["n_shared_experts"] or 0)
    return int(cfg.get("num_shared_experts", 0) or 0)


def _router_weight_suffix(cfg: dict) -> str:
    return str(cfg.get("_router_weight_suffix", "gate.weight"))


def _router_bias_suffix(cfg: dict) -> str:
    return str(cfg.get(
        "_router_bias_suffix", "gate.e_score_correction_bias"))


def _shared_expert_stem(cfg: dict) -> str:
    return str(cfg.get("_shared_expert_stem", "shared_experts"))


def _canonical_non_expert_name(wname: str, cfg: dict) -> str:
    """Map checkpoint-specific HY-V3 names to the graph's stable ABI."""
    wname = wname.replace(
        ".mlp." + _router_weight_suffix(cfg), ".mlp.gate.weight")
    wname = wname.replace(
        "." + _router_bias_suffix(cfg),
        ".gate.e_score_correction_bias")
    mtp_head_norm = str(cfg.get(
        "_mtp_head_norm_suffix", "shared_head.norm.weight"))
    wname = wname.replace(
        "." + mtp_head_norm, ".shared_head.norm.weight")
    return wname.replace(
        ".mlp." + _shared_expert_stem(cfg) + ".",
        ".mlp.shared_experts.")


def _router_score_func(cfg: dict) -> int:
    scoring_func = cfg.get("scoring_func")
    if scoring_func == "sigmoid":
        return ROUTER_SCORE_SIGMOID
    if scoring_func == "softmax":
        return ROUTER_SCORE_SOFTMAX
    return ROUTER_SCORE_SIGMOID if _has_router_bias(cfg) else ROUTER_SCORE_SOFTMAX


def _mtp_quantization(quant: str, override: str | None = None) -> str:
    """Resolve the draft-block precision independently of the target."""
    if override is not None:
        return _canonical_quant(override)
    if quant.startswith("w4mixg"):
        return "w8g" + quant[len("w4mixg"):]
    if quant.startswith("w4g"):
        return "w8g" + quant[len("w4g"):]
    return quant


def _prepare_config_for_weights(model_dir: Path, cfg: dict) -> dict:
    cfg = dict(cfg)
    weight_names = set(_weight_map(model_dir))
    has_q_norm = "model.layers.0.self_attn.q_norm.weight" in weight_names
    has_q_layernorm = "model.layers.0.self_attn.q_layernorm.weight" in weight_names
    if has_q_norm:
        cfg["_attn_norm_kind"] = "q_norm"
    elif has_q_layernorm:
        cfg["_attn_norm_kind"] = "q_layernorm"
    else:
        raise KeyError("missing Q/K attention norm weights")
    first_moe_layer = int(cfg.get("first_k_dense_replace", 0))
    mlp = f"model.layers.{first_moe_layer}.mlp"
    router_candidates = ("gate.weight", "router.gate.weight")
    cfg["_router_weight_suffix"] = next(
        (suffix for suffix in router_candidates
         if f"{mlp}.{suffix}" in weight_names), router_candidates[0])
    bias_candidates = ("gate.e_score_correction_bias", "expert_bias")
    matched_bias = next(
        (suffix for suffix in bias_candidates
         if f"{mlp}.{suffix}" in weight_names), None)
    cfg["_has_router_correction_bias"] = matched_bias is not None
    if matched_bias is not None:
        cfg["_router_bias_suffix"] = matched_bias

    shared_candidates = ("shared_experts", "shared_mlp")
    cfg["_shared_expert_stem"] = next(
        (stem for stem in shared_candidates
         if f"{mlp}.{stem}.gate_proj.weight" in weight_names),
        shared_candidates[0])

    mtp_idx = int(cfg.get("num_hidden_layers", 0))
    mtp_norm_candidates = (
        "shared_head.norm.weight", "final_layernorm.weight")
    cfg["_mtp_head_norm_suffix"] = next(
        (suffix for suffix in mtp_norm_candidates
         if f"model.layers.{mtp_idx}.{suffix}" in weight_names),
        mtp_norm_candidates[0])

    # HY-V3 uses Qwen-style full attention, but names several MoE config
    # fields differently. Normalize only the semantic aliases consumed by the
    # graph builder; retain model_type so package metadata remains accurate.
    if cfg.get("model_type") == "hy_v3":
        if not bool(cfg.get("moe_router_use_sigmoid", True)):
            raise ValueError("HY-V3 softmax routing is not supported")
        cfg["scoring_func"] = "sigmoid"
        cfg["norm_topk_prob"] = bool(cfg.get("route_norm", True))
        cfg["routed_scaling_factor"] = float(
            cfg.get("router_scaling_factor", 1.0))
        cfg["n_shared_experts"] = int(
            cfg.get("num_shared_experts", 0) or 0)
    return cfg


def _as_fp32(dtype_str: str, arr: np.ndarray) -> np.ndarray:
    if dtype_str == "BF16":
        as_u32 = arr.astype(np.uint32) << 16
        return as_u32.view(np.float32)
    if dtype_str == "F16":
        return arr.astype(np.float32)
    if dtype_str == "F32":
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
    raise ValueError(f"unsupported safetensors dtype: {dtype_str}")


def _as_fp16(dtype_str: str, arr: np.ndarray) -> np.ndarray:
    if dtype_str == "F16":
        return arr.astype(np.float16) if arr.dtype != np.float16 else arr
    return _as_fp32(dtype_str, arr).astype(np.float16)


def _weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    files = sorted(model_dir.glob("model-*.safetensors"))
    if not files:
        files = sorted(model_dir.glob("model.safetensors-*.safetensors"))
    if not files:
        files = list(model_dir.glob("model.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors file in {model_dir}")

    out: dict[str, str] = {}
    for path in files:
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
        for name in header:
            if name != "__metadata__":
                out[name] = path.name
    return out


def _selected_safetensors(model_dir: Path, wanted: set[str]):
    weight_map = _weight_map(model_dir)
    missing = sorted(wanted - set(weight_map))
    if missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"missing required tensors: {preview}")

    by_file: dict[str, list[str]] = defaultdict(list)
    for name in sorted(wanted):
        by_file[weight_map[name]].append(name)

    for fname, names in sorted(by_file.items()):
        path = model_dir / fname
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len).decode("utf-8"))
            data_base = 8 + header_len
            for name in names:
                meta = header[name]
                dtype_str = meta["dtype"]
                shape = meta["shape"]
                begin, end = meta["data_offsets"]
                f.seek(data_base + begin)
                raw = f.read(end - begin)
                np_dtype = {
                    "F32": np.float32,
                    "F16": np.float16,
                    "BF16": np.uint16,
                }[dtype_str]
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                yield name, dtype_str, arr


def _write_maybe_quantized(path: str, data: np.ndarray, quant: str,
                           quantizable: bool, raw_name: str,
                           quant_counts: dict[str, int]):
    qspec = (
        _quant_spec(quant, data.shape[1], raw_name)
        if quantizable and data.ndim == 2 else None
    )
    if qspec is not None:
        quant_kind, group_size = qspec
        if write_quantized_weight_file_cpp(
            path, data, quant_kind, group_size, required=(quant_kind == "w4")
        ):
            quant_counts[quant_kind] += 1
            return
        if quant_kind == "w8":
            q, scales, gs, ng = quantize_weight_w8_group(data, group_size)
            _write_weight_file(path, q, scales=scales,
                               group_size=gs, num_groups=ng)
            quant_counts[quant_kind] += 1
            return
        raise ValueError(f"unsupported quant kind for {raw_name}: {quant_kind}")

    _write_weight_file(path, data)


def _non_expert_weight_names(cfg: dict, num_layers: int,
                             available: set[str]) -> set[str]:
    names = {
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    q_norm_name, k_norm_name = _norm_suffixes(cfg)
    has_router_bias = _has_router_bias(cfg)
    for i in range(num_layers):
        pfx = f"model.layers.{i}"
        attn = f"{pfx}.self_attn"
        names.update({
            f"{pfx}.input_layernorm.weight",
            f"{pfx}.post_attention_layernorm.weight",
            f"{attn}.q_proj.weight",
            f"{attn}.k_proj.weight",
            f"{attn}.v_proj.weight",
            f"{attn}.o_proj.weight",
            f"{attn}.{q_norm_name}.weight",
            f"{attn}.{k_norm_name}.weight",
        })
        mlp = f"{pfx}.mlp"
        if i < dense_until:
            names.update({
                f"{mlp}.gate_proj.weight",
                f"{mlp}.up_proj.weight",
                f"{mlp}.down_proj.weight",
            })
        else:
            names.add(f"{mlp}.{_router_weight_suffix(cfg)}")
            if has_router_bias:
                names.add(f"{mlp}.{_router_bias_suffix(cfg)}")
            if _num_shared_experts(cfg) > 0:
                shared = _shared_expert_stem(cfg)
                names.update({
                    f"{mlp}.{shared}.gate_proj.weight",
                    f"{mlp}.{shared}.up_proj.weight",
                    f"{mlp}.{shared}.down_proj.weight",
                })
    return {name for name in names if name in available}


def _expert_weight_names(layer_idx: int, num_experts: int) -> set[str]:
    mlp = f"model.layers.{layer_idx}.mlp"
    names: set[str] = set()
    for e in range(num_experts):
        ep = f"{mlp}.experts.{e}"
        names.update({
            f"{ep}.gate_proj.weight",
            f"{ep}.up_proj.weight",
            f"{ep}.down_proj.weight",
        })
    return names


def _model_name_from_config(cfg: dict, fallback: str) -> str:
    for key in ("model_name", "_name_or_path", "name_or_path",
                "base_model_name_or_path"):
        value = cfg.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip().rstrip("/\\")).name
    return fallback


def _fallback_model_name(model_dir: Path, cfg: dict, num_layers: int) -> str:
    if cfg.get("model_type") in ("qwen3_moe", "hy_v3") and model_dir.name:
        return model_dir.name
    return f"Qwen3-MoE-{num_layers}L"


def export_weights(model_dir: Path, weights_dir: str, cfg: dict,
                   num_layers: int, quant: str = "fp16",
                   mtp_layers: int = 0,
                   mtp_quant: str | None = None):
    os.makedirs(weights_dir, exist_ok=True)
    quant = _canonical_quant(quant)
    quant_counts = {"w4": 0, "w8": 0}
    hidden_size = int(cfg["hidden_size"])
    intermediate = int(cfg["intermediate_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    num_experts = _num_experts(cfg)
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    n_shared = _num_shared_experts(cfg)
    weight_map = _weight_map(model_dir)
    qkv_parts: dict[int, dict[str, np.ndarray]] = {}
    dense_gate_up_parts: dict[int, dict[str, np.ndarray]] = {}
    shared_gate_up_parts: dict[int, dict[str, np.ndarray]] = {}
    mtp_prefix = f"model.layers.{num_layers}." if mtp_layers else ""
    resolved_mtp_quant = _mtp_quantization(quant, mtp_quant)

    def effective_quant(raw_name: str) -> str:
        if not mtp_prefix or not raw_name.startswith(mtp_prefix):
            return quant
        # Draft-head errors directly reduce the number of target tokens saved.
        # Keep a W4 target package compact, but give its single MTP block W8
        # weights so quantization noise does not erase speculative acceptance.
        return resolved_mtp_quant

    def save(name: str, data: np.ndarray, quantizable: bool = False,
             raw_name: str = ""):
        _write_maybe_quantized(
            os.path.join(weights_dir, f"{name}.weights"),
            data, effective_quant(raw_name or name), quantizable,
            raw_name or name, quant_counts)

    print("  Exporting non-expert weights...")
    total_layers = num_layers + mtp_layers
    non_expert_names = _non_expert_weight_names(
        cfg, total_layers, set(weight_map))
    if mtp_layers:
        mtp_idx = num_layers
        non_expert_names.update({
            f"model.layers.{mtp_idx}.eh_proj.weight",
            f"model.layers.{mtp_idx}.enorm.weight",
            f"model.layers.{mtp_idx}.hnorm.weight",
            f"model.layers.{mtp_idx}."
            f"{cfg.get('_mtp_head_norm_suffix', 'shared_head.norm.weight')}",
        })
    for wname, dtype_str, wdata in _selected_safetensors(
            model_dir, non_expert_names):
        if wname == "model.embed_tokens.weight":
            save("embed_tokens", _as_fp16(dtype_str, wdata))
            continue
        if wname == "lm_head.weight":
            save("lm_head", _as_fp16(dtype_str, wdata),
                 quantizable=True, raw_name=wname)
            continue
        if wname == "model.norm.weight":
            save("final_norm", _as_fp32(dtype_str, wdata))
            continue
        if "norm" in wname.lower() or "layernorm" in wname.lower():
            canonical = _canonical_non_expert_name(wname, cfg)
            save(canonical.replace(".", "_"),
                 _as_fp32(dtype_str, wdata))
            continue
        if wname.endswith("." + _router_bias_suffix(cfg)):
            canonical = _canonical_non_expert_name(wname, cfg)
            save(canonical.replace(".", "_"),
                 _as_fp32(dtype_str, wdata))
            continue
        if wname.endswith(".mlp." + _router_weight_suffix(cfg)):
            canonical = _canonical_non_expert_name(wname, cfg)
            save(canonical.replace(".", "_"), _as_fp16(dtype_str, wdata),
                 quantizable=False, raw_name=wname)
            continue
        qkv_match = re.match(
            r"^model\.layers\.(\d+)\.self_attn\."
            r"(q_proj|k_proj|v_proj)\.weight$",
            wname)
        if qkv_match:
            layer_idx = int(qkv_match.group(1))
            kind = qkv_match.group(2)
            parts = qkv_parts.setdefault(layer_idx, {})
            parts[kind] = _as_fp16(dtype_str, wdata)
            if len(parts) == 3:
                merged = np.concatenate(
                    [parts["q_proj"], parts["k_proj"], parts["v_proj"]],
                    axis=0)
                save(
                    f"model_layers_{layer_idx}_self_attn_qkv_proj_weight",
                    merged, quantizable=True,
                    raw_name=(
                        f"model.layers.{layer_idx}."
                        "self_attn.qkv_proj.weight"))
                del qkv_parts[layer_idx]
            continue
        dense_match = re.match(
            r"^model\.layers\.(\d+)\.mlp\."
            r"(gate_proj|up_proj)\.weight$",
            wname)
        if dense_match and int(dense_match.group(1)) < dense_until:
            layer_idx = int(dense_match.group(1))
            kind = dense_match.group(2)
            parts = dense_gate_up_parts.setdefault(layer_idx, {})
            parts[kind] = _as_fp16(dtype_str, wdata)
            if len(parts) == 2:
                merged = np.concatenate(
                    [parts["gate_proj"], parts["up_proj"]], axis=0)
                save(
                    f"model_layers_{layer_idx}_mlp_gate_up_proj_weight",
                    merged, quantizable=True,
                    raw_name=(
                        f"model.layers.{layer_idx}."
                        "mlp.gate_up_proj.weight"))
                del dense_gate_up_parts[layer_idx]
            continue
        shared_match = re.match(
            r"^model\.layers\.(\d+)\.mlp\."
            + re.escape(_shared_expert_stem(cfg))
            + r"\.(gate_proj|up_proj)\.weight$",
            wname)
        if shared_match:
            layer_idx = int(shared_match.group(1))
            kind = shared_match.group(2)
            parts = shared_gate_up_parts.setdefault(layer_idx, {})
            parts[kind] = _as_fp16(dtype_str, wdata)
            if len(parts) == 2:
                merged = np.concatenate(
                    [parts["gate_proj"], parts["up_proj"]], axis=0)
                save(
                    f"model_layers_{layer_idx}_mlp_"
                    "shared_experts_gate_up_proj_weight",
                    merged, quantizable=True,
                    raw_name=(
                        f"model.layers.{layer_idx}.mlp."
                        "shared_experts.gate_up_proj.weight"))
                del shared_gate_up_parts[layer_idx]
            continue
        d = _as_fp16(dtype_str, wdata)
        canonical = _canonical_non_expert_name(wname, cfg)
        save(canonical.replace(".", "_"), d,
             quantizable=True, raw_name=wname)

    if qkv_parts:
        raise KeyError(
            "incomplete Q/K/V projection sets for layers: "
            + ", ".join(str(i) for i in sorted(qkv_parts)))
    if dense_gate_up_parts:
        raise KeyError(
            "incomplete dense gate/up projection sets for layers: "
            + ", ".join(str(i) for i in sorted(dense_gate_up_parts)))
    if shared_gate_up_parts:
        raise KeyError(
            "incomplete shared-expert gate/up projection sets for layers: "
            + ", ".join(str(i) for i in sorted(shared_gate_up_parts)))

    for layer_idx in range(dense_until, total_layers):
        print(f"  Exporting MoE experts layer {layer_idx}/{total_layers - 1}...")
        gate_up = np.empty((num_experts * 2 * moe_intermediate, hidden_size),
                           dtype=np.float16)
        down = np.empty((num_experts * hidden_size, moe_intermediate),
                        dtype=np.float16)
        for wname, dtype_str, wdata in _selected_safetensors(
                model_dir, _expert_weight_names(layer_idx, num_experts)):
            parts = wname.split(".")
            expert_idx = int(parts[5])
            kind = parts[6]
            d = _as_fp16(dtype_str, wdata)
            if kind == "gate_proj":
                row = expert_idx * 2 * moe_intermediate
                gate_up[row:row + moe_intermediate, :] = d
            elif kind == "up_proj":
                row = expert_idx * 2 * moe_intermediate + moe_intermediate
                gate_up[row:row + moe_intermediate, :] = d
            elif kind == "down_proj":
                row = expert_idx * hidden_size
                down[row:row + hidden_size, :] = d

        mlp_name = f"model_layers_{layer_idx}_mlp"
        save(f"{mlp_name}_experts_gate_up_proj", gate_up,
             quantizable=True,
             raw_name=f"model.layers.{layer_idx}.mlp.experts.gate_up")
        save(f"{mlp_name}_experts_down_proj", down,
             quantizable=True,
             raw_name=f"model.layers.{layer_idx}.mlp.experts.down")

    if n_shared > 0:
        print("  Shared expert gate/up projections were fused for the graph path.")

    if quant != "fp16":
        print(f"  Quantized tensors: W4={quant_counts['w4']} W8={quant_counts['w8']}")


def build_graph(weights_dir: str, cfg: dict, seq_len: int = 1,
                n_ctx: int = 16384, is_prefill: bool = False) -> GraphBuilder:
    g = GraphBuilder()
    hidden_size = int(cfg["hidden_size"])
    num_layers = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))
    intermediate = int(cfg["intermediate_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    num_experts = _num_experts(cfg)
    top_k = int(cfg["num_experts_per_tok"])
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    n_shared = _num_shared_experts(cfg)
    shared_intermediate = n_shared * moe_intermediate
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_interleave = bool(cfg.get("rope_interleave", False))
    rope_dim = head_dim
    vocab_size = int(cfg["vocab_size"])
    router_score_func = _router_score_func(cfg)

    print(f"Qwen3-MoE graph: seq_len={seq_len}, layers={num_layers}, "
          f"dense_until={dense_until}, experts={num_experts}, top_k={top_k}, "
          f"moe_intermediate={moe_intermediate}")

    g.set_model_config(
        rope_dim=rope_dim,
        rope_theta=rope_theta,
        hidden_size=hidden_size,
        num_layers=num_layers,
        vocab_size=vocab_size,
        model_type=str(cfg.get("model_type", "qwen3_moe")),
    )

    embed_shape = (vocab_size, hidden_size)
    g.weight(os.path.join(weights_dir, "embed_tokens.weights"),
             embed_shape, Precision.FP16)
    g.weight(os.path.join(weights_dir, "lm_head.weights"),
             embed_shape, Precision.FP16)

    if is_prefill:
        from transpile import DimExpr
        CONST = DimExpr.const()
        SEQ_DIM = DimExpr.seq()
        hidden_dyn = (CONST, SEQ_DIM, CONST, CONST)
        mask_dyn = (CONST, SEQ_DIM, CONST, CONST)
        cos_dyn = (CONST, SEQ_DIM, CONST, CONST)
        sin_dyn = (CONST, SEQ_DIM, CONST, CONST)
    else:
        hidden_dyn = mask_dyn = cos_dyn = sin_dyn = None

    hidden = g.input("hidden", (hidden_size, seq_len), dynamic=hidden_dyn)
    mask = g.input("mask", (1, seq_len), dynamic=mask_dyn)
    cos = g.input("cos", (rope_dim // 2, seq_len), dynamic=cos_dyn)
    sin = g.input("sin", (rope_dim // 2, seq_len), dynamic=sin_dyn)

    caches = []
    for i in range(num_layers):
        ck = g.input(f"cache_k{i}", (head_dim, n_ctx, num_kv_heads),
                     prec=Precision.FP16)
        cv = g.input(f"cache_v{i}", (head_dim, n_ctx, num_kv_heads),
                     prec=Precision.FP16)
        caches.append((ck, cv))

    input_norm_weights = [
        g.weight(
            os.path.join(
                weights_dir,
                f"model_layers_{i}_input_layernorm_weight.weights"),
            (hidden_size,), Precision.FP32)
        for i in range(num_layers)
    ]
    post_norm_weights = [
        g.weight(
            os.path.join(
                weights_dir,
                f"model_layers_{i}_post_attention_layernorm_weight.weights"),
            (hidden_size,), Precision.FP32)
        for i in range(num_layers)
    ]
    final_norm_weight = g.weight(
        os.path.join(weights_dir, "final_norm.weights"),
        (hidden_size,), Precision.FP32)

    x = hidden
    x_normed = g.rms_norm(x, input_norm_weights[0], eps=eps)
    for i in range(num_layers):
        ck, cv = caches[i]
        next_norm_weight = (
            input_norm_weights[i + 1]
            if i + 1 < num_layers else final_norm_weight
        )
        x, x_normed = _build_layer(
            g, x, x_normed, post_norm_weights[i], next_norm_weight,
            i, weights_dir, cos, sin, mask, ck, cv, eps, seq_len,
            num_heads, num_kv_heads, head_dim, hidden_size, intermediate,
            moe_intermediate, num_experts, top_k, dense_until,
            shared_intermediate, n_shared, _has_router_bias(cfg),
            router_score_func,
            bool(cfg.get("norm_topk_prob", True)),
            int(cfg.get("n_group", 1)),
            int(cfg.get("topk_group", 1)),
            float(cfg.get("routed_scaling_factor", 1.0)),
            str(cfg.get("_attn_norm_kind", "q_layernorm")),
            rope_interleave,
            is_prefill=is_prefill)

    x = x_normed

    print(f"  Total: {len(g._nodes)} nodes")
    return g


def build_mtp_graph(weights_dir: str, cfg: dict, seq_len: int = 256,
                    n_ctx: int = 16384) -> GraphBuilder:
    """Build the single Qwen3-style next-token prediction decoder block.

    The MTP block consumes the embedding of token t and the target decoder's
    final hidden state at t-1. It owns an independent KV cache; the runtime
    periodically replaces speculative hidden inputs with verified target
    hidden states to keep that cache exact.
    """
    from transpile import DimExpr

    g = GraphBuilder()
    hidden_size = int(cfg["hidden_size"])
    layer_idx = int(cfg["num_hidden_layers"])
    num_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    head_dim = int(cfg.get("head_dim", hidden_size // num_heads))
    intermediate = int(cfg["intermediate_size"])
    moe_intermediate = int(cfg["moe_intermediate_size"])
    num_experts = _num_experts(cfg)
    top_k = int(cfg["num_experts_per_tok"])
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    n_shared = _num_shared_experts(cfg)
    shared_intermediate = n_shared * moe_intermediate
    eps = float(cfg.get("rms_norm_eps", 1e-6))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_interleave = bool(cfg.get("rope_interleave", False))

    g.set_model_config(
        rope_dim=head_dim, rope_theta=rope_theta,
        hidden_size=hidden_size, num_layers=1,
        vocab_size=int(cfg["vocab_size"]), model_type="qwen3_moe_mtp")

    const = DimExpr.const()
    seq = DimExpr.seq()
    hidden_dyn = (const, seq, const, const)
    mask_dyn = (const, seq, const, const)
    rope_dyn = (const, seq, const, const)
    target_hidden = g.input(
        "target_hidden", (hidden_size, seq_len), dynamic=hidden_dyn)
    token_hidden = g.input(
        "hidden", (hidden_size, seq_len), dynamic=hidden_dyn)
    mask = g.input("mask", (1, seq_len), dynamic=mask_dyn)
    cos = g.input("cos", (head_dim // 2, seq_len), dynamic=rope_dyn)
    sin = g.input("sin", (head_dim // 2, seq_len), dynamic=rope_dyn)
    ck = g.input("cache_k0", (head_dim, n_ctx, num_kv_heads),
                 prec=Precision.FP16)
    cv = g.input("cache_v0", (head_dim, n_ctx, num_kv_heads),
                 prec=Precision.FP16)

    pfx = f"model_layers_{layer_idx}"
    hnorm = g.weight(
        os.path.join(weights_dir, f"{pfx}_hnorm_weight.weights"),
        (hidden_size,), Precision.FP32)
    enorm = g.weight(
        os.path.join(weights_dir, f"{pfx}_enorm_weight.weights"),
        (hidden_size,), Precision.FP32)
    eh_proj = g.weight(
        os.path.join(weights_dir, f"{pfx}_eh_proj_weight.weights"),
        (hidden_size, 2 * hidden_size), Precision.FP16)
    input_norm = g.weight(
        os.path.join(weights_dir, f"{pfx}_input_layernorm_weight.weights"),
        (hidden_size,), Precision.FP32)
    post_norm = g.weight(
        os.path.join(
            weights_dir, f"{pfx}_post_attention_layernorm_weight.weights"),
        (hidden_size,), Precision.FP32)
    head_norm = g.weight(
        os.path.join(weights_dir, f"{pfx}_shared_head_norm_weight.weights"),
        (hidden_size,), Precision.FP32)

    h = g.rms_norm(target_hidden, hnorm, eps=eps)
    e = g.rms_norm(token_hidden, enorm, eps=eps)
    x = g.matmul(g.concat([e, h], dim=0), eh_proj)
    x_normed = g.rms_norm(x, input_norm, eps=eps)
    _, x_normed = _build_layer(
        g, x, x_normed, post_norm, head_norm, layer_idx, weights_dir,
        cos, sin, mask, ck, cv, eps, seq_len, num_heads, num_kv_heads,
        head_dim, hidden_size, intermediate, moe_intermediate, num_experts,
        top_k, dense_until, shared_intermediate, n_shared,
        _has_router_bias(cfg), _router_score_func(cfg),
        bool(cfg.get("norm_topk_prob", True)), int(cfg.get("n_group", 1)),
        int(cfg.get("topk_group", 1)),
        float(cfg.get("routed_scaling_factor", 1.0)),
        str(cfg.get("_attn_norm_kind", "q_layernorm")), rope_interleave,
        is_prefill=True)
    return g


def _build_layer(g: GraphBuilder, x: int, x_normed: int,
                 post_norm_weight: int, next_norm_weight: int,
                 layer_idx: int, weights_dir: str,
                 cos: int, sin: int, mask: int, ck_in: int, cv_in: int,
                 eps: float, seq_len: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, hidden_size: int, intermediate: int,
                 moe_intermediate: int, num_experts: int, top_k: int,
                 dense_until: int, shared_intermediate: int, n_shared: int,
                 has_router_bias: bool, router_score_func: int,
                 norm_topk_prob: bool, n_group: int, topk_group: int,
                 routed_scaling_factor: float, attn_norm_kind: str,
                 rope_interleave: bool, is_prefill: bool = False) -> int:
    pfx = f"model_layers_{layer_idx}"

    attn_out = _build_attention(
        g, x_normed, layer_idx, weights_dir, cos, sin, mask, ck_in, cv_in,
        eps, seq_len, num_heads, num_kv_heads, head_dim, hidden_size,
        attn_norm_kind, rope_interleave, is_prefill=is_prefill)
    x_normed2 = g.add_rms_norm(
        x, attn_out, post_norm_weight, eps=eps)

    mlp_pfx = f"{pfx}_mlp"
    if layer_idx < dense_until:
        w_gate_up = g.weight(
            os.path.join(
                weights_dir, f"{mlp_pfx}_gate_up_proj_weight.weights"),
            (2 * intermediate, hidden_size), Precision.FP16)
        w_down = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_down_proj_weight.weights"),
                          (hidden_size, intermediate), Precision.FP16)
        mlp_hidden = g.swiglu(g.matmul(x_normed2, w_gate_up))
        mlp_out = g.matmul(mlp_hidden, w_down)
        next_x_normed = g.add_rms_norm(
            x, mlp_out, next_norm_weight, eps=eps)
        return x, next_x_normed

    w_router = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_weight.weights"),
                        (num_experts, hidden_size), Precision.FP16)
    w_bias = None
    if has_router_bias:
        w_bias = g.weight(os.path.join(weights_dir, f"{mlp_pfx}_gate_e_score_correction_bias.weights"),
                          (num_experts,), Precision.FP32)
    w_experts_gate_up = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_experts_gate_up_proj.weights"),
        (num_experts * 2 * moe_intermediate, hidden_size), Precision.FP16)
    w_experts_down = g.weight(
        os.path.join(weights_dir, f"{mlp_pfx}_experts_down_proj.weights"),
        (num_experts * hidden_size, moe_intermediate), Precision.FP16)

    has_shared = n_shared > 0
    mlp_out = g.moe(
        x_normed2, w_router, w_experts_gate_up, w_experts_down,
        x_normed2, x_normed2, x_normed2, None,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=moe_intermediate,
        shared_intermediate_size=0,
        router_bias=w_bias,
        router_score_func=router_score_func,
        norm_topk_prob=norm_topk_prob,
        has_shared_expert=False,
        shared_expert_has_gate=False,
        n_group=n_group,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor)
    if has_shared:
        w_shared_gate_up = g.weight(
            os.path.join(
                weights_dir,
                f"{mlp_pfx}_shared_experts_gate_up_proj_weight.weights"),
            (2 * shared_intermediate, hidden_size), Precision.FP16)
        w_shared_down = g.weight(
            os.path.join(
                weights_dir,
                f"{mlp_pfx}_shared_experts_down_proj_weight.weights"),
            (hidden_size, shared_intermediate), Precision.FP16)
        shared_hidden = g.swiglu(g.matmul(x_normed2, w_shared_gate_up))
        shared_out = g.matmul(shared_hidden, w_shared_down)
        mlp_out = g.add(mlp_out, shared_out)
    next_x_normed = g.add_rms_norm(
        x, mlp_out, next_norm_weight, eps=eps)
    return x, next_x_normed


def _build_attention(g: GraphBuilder, x: int, layer_idx: int, weights_dir: str,
                     cos: int, sin: int, mask: int, ck_in: int, cv_in: int,
                     eps: float, seq_len: int, num_heads: int,
                     num_kv_heads: int, head_dim: int, hidden_size: int,
                     attn_norm_kind: str, rope_interleave: bool,
                     is_prefill: bool = False) -> int:
    from transpile import SEQ as SEQ_SYMBOL

    _S = SEQ_SYMBOL.bind(seq_len) if is_prefill else seq_len
    pfx = f"model_layers_{layer_idx}_self_attn"

    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    w_qkv = g.weight(
        os.path.join(weights_dir, f"{pfx}_qkv_proj_weight.weights"),
        (q_dim + 2 * kv_dim, hidden_size), Precision.FP16)
    qkv = g.matmul(x, w_qkv)
    query, k, v = g.slice(qkv, [q_dim, kv_dim, kv_dim], dim=0)

    q_norm_name, k_norm_name = _norm_suffixes({
        "_attn_norm_kind": attn_norm_kind,
    })

    w_qn = g.weight(os.path.join(weights_dir, f"{pfx}_{q_norm_name}_weight.weights"),
                    (head_dim,), Precision.FP32)
    w_kn = g.weight(os.path.join(weights_dir, f"{pfx}_{k_norm_name}_weight.weights"),
                    (head_dim,), Precision.FP32)

    query = g.reshape(query, (head_dim, num_heads, _S))
    query = g.permute(query, (0, 2, 1, 3))
    query = g.reshape(query, (head_dim, num_heads * _S))
    k = g.reshape(k, (head_dim, num_kv_heads, _S))
    k = g.permute(k, (0, 2, 1, 3))
    k = g.reshape(k, (head_dim, num_kv_heads * _S))
    qk = g.qk_rms_norm_rope(
        query, k, w_qn, w_kn, cos, sin, _S, num_heads, num_kv_heads,
        rope_dim=head_dim, interleave=rope_interleave, eps=eps)
    query, k = g.slice(qk, [num_heads, num_kv_heads], dim=2)

    v = g.reshape(v, (head_dim, num_kv_heads, _S))
    v = g.permute(v, (0, 2, 1, 3))

    attn, _, _ = g.sdpa(
        query, k, v, mask, ck_in, cv_in,
        kv_cache=2, causal=True, scale=head_dim ** -0.5,
        num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, v_head_dim=head_dim)

    attn = g.permute(attn, (0, 2, 1, 3))
    attn = g.contiguous(attn)
    attn = g.reshape(attn, (num_heads * head_dim, _S))

    w_o = g.weight(os.path.join(weights_dir, f"{pfx}_o_proj_weight.weights"),
                   (hidden_size, num_heads * head_dim), Precision.FP16)
    return g.matmul(attn, w_o)


def _moe_expert_storage_metadata(weights_dir: str, cfg: dict,
                                 num_layers: int) -> dict:
    hidden_size = int(cfg["hidden_size"])
    intermediate = int(cfg["moe_intermediate_size"])
    num_experts = _num_experts(cfg)
    dense_until = int(cfg.get("first_k_dense_replace", 0))
    layers = []
    for layer_idx in range(dense_until, num_layers):
        mlp_pfx = f"model_layers_{layer_idx}_mlp"
        layers.append({
            "layer": layer_idx,
            "num_experts": num_experts,
            "gate_up": {
                "weight": os.path.join(weights_dir, f"{mlp_pfx}_experts_gate_up_proj.weights"),
                "rows_per_expert": 2 * intermediate,
                "cols": hidden_size,
                "logical_shape": [num_experts, 2 * intermediate, hidden_size],
            },
            "down": {
                "weight": os.path.join(weights_dir, f"{mlp_pfx}_experts_down_proj.weights"),
                "rows_per_expert": hidden_size,
                "cols": intermediate,
                "logical_shape": [num_experts, hidden_size, intermediate],
            },
        })
    return {
        "version": 1,
        "layout": "aggregate_rows_v1",
        "num_experts": num_experts,
        "layers": layers,
    }


def convert_qwen3_moe(model_dir: str, output_path: str,
                      num_layers: int | None = None,
                      prefill_seq_len: int = 256,
                      n_ctx: int = 16384,
                      quant: str = "fp16",
                      mtp_quant: str | None = None):
    model_dir = Path(model_dir)
    quant = _canonical_quant(quant)
    resolved_mtp_quant = _mtp_quantization(quant, mtp_quant)
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)
    cfg = _prepare_config_for_weights(model_dir, cfg)

    config_num_layers = int(cfg["num_hidden_layers"])
    if num_layers is None:
        num_layers = config_num_layers
    if num_layers <= 0 or num_layers > config_num_layers:
        raise ValueError(f"num_layers must be in [1, {config_num_layers}], got {num_layers}")
    if num_layers != config_num_layers:
        print(f"Debug: truncating Qwen3-MoE layers to {num_layers}/{config_num_layers}")
    cfg["num_hidden_layers"] = num_layers
    mtp_layers = int(cfg.get("num_nextn_predict_layers", 0) or 0)
    if num_layers != config_num_layers:
        mtp_layers = 0
    if mtp_layers not in (0, 1):
        raise ValueError("only one Qwen3-style MTP layer is currently supported")
    if mtp_layers:
        available = set(_weight_map(model_dir))
        required = {
            f"model.layers.{num_layers}.eh_proj.weight",
            f"model.layers.{num_layers}.enorm.weight",
            f"model.layers.{num_layers}.hnorm.weight",
            f"model.layers.{num_layers}."
            f"{cfg.get('_mtp_head_norm_suffix', 'shared_head.norm.weight')}",
        }
        if not required.issubset(available):
            print("MTP metadata is present but MTP tensors are incomplete; "
                  "converting the target model only")
            mtp_layers = 0

    tmp_dir = tempfile.mkdtemp(prefix="mollm_qwen3_moe_weights_")
    weights_dir = tmp_dir
    weights_rel = "."
    try:
        print("Exporting selected Qwen3-MoE weights...")
        export_weights(model_dir, weights_dir, cfg, num_layers, quant=quant,
                       mtp_layers=mtp_layers,
                       mtp_quant=resolved_mtp_quant)

        print(f"\nBuilding prefill graph (seq_len={prefill_seq_len})...")
        g_prefill = build_graph(weights_rel, cfg, seq_len=prefill_seq_len,
                                n_ctx=n_ctx, is_prefill=True)

        print("\nBuilding decode graph (seq_len=1)...")
        g_decode = build_graph(weights_rel, cfg, seq_len=1, n_ctx=n_ctx)

        g_mtp = None
        if mtp_layers:
            print("\nBuilding MTP graph...")
            g_mtp = build_mtp_graph(
                weights_rel, cfg, seq_len=prefill_seq_len, n_ctx=n_ctx)

        fallback_model_name = _fallback_model_name(model_dir, cfg, num_layers)
        metadata = {
            "model_name": _model_name_from_config(cfg, fallback_model_name),
            "architecture": (
                "hy-v3" if cfg.get("model_type") == "hy_v3"
                else "qwen3-moe"),
            "num_layers": num_layers,
            "hidden_size": cfg["hidden_size"],
            "num_heads": cfg["num_attention_heads"],
            "num_kv_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"],
            "prefill_seq_len": prefill_seq_len,
            "n_ctx": n_ctx,
            "vocab_size": cfg["vocab_size"],
            "num_experts": _num_experts(cfg),
            "num_experts_per_tok": cfg["num_experts_per_tok"],
            "moe_intermediate_size": cfg["moe_intermediate_size"],
            "first_k_dense_replace": cfg.get("first_k_dense_replace", 0),
            "n_shared_experts": _num_shared_experts(cfg),
            "moe_expert_storage": _moe_expert_storage_metadata(
                weights_rel, cfg, num_layers + mtp_layers),
            "num_nextn_predict_layers": mtp_layers,
            "mtp_quantization": (
                resolved_mtp_quant if mtp_layers else "none"),
            "quantization": quant,
        }

        print(f"\nPacking {output_path}...")
        save_package(output_path, g_prefill, g_decode, weights_dir, metadata,
                     tokenizer_path=str(model_dir / "tokenizer.json"),
                     jinja_path=str(model_dir / "chat_template.jinja"),
                     g_mtp=g_mtp,
                     remove_weight_files=True)
        print(f"\nDone! Output: {output_path}")
    finally:
        shutil.rmtree(tmp_dir)


def convert_hy_v3(model_dir: str, output_path: str,
                  num_layers: int | None = None,
                  prefill_seq_len: int = 256,
                  n_ctx: int = 16384,
                  quant: str = "fp16"):
    """Convert an official HY-V3 checkpoint using the shared MoE backend."""
    return convert_qwen3_moe(
        model_dir, output_path, num_layers=num_layers,
        prefill_seq_len=prefill_seq_len, n_ctx=n_ctx, quant=quant)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Qwen3-MoE-compatible models to .mollm")
    parser.add_argument("model_dir")
    parser.add_argument("output")
    parser.add_argument("quant", nargs="?", default="fp16")
    parser.add_argument("--layers", type=int, default=None,
                        help="debug-only layer truncation")
    parser.add_argument("--prefill-seq-len", type=int, default=256,
                        help="prefill graph chunk length")
    parser.add_argument(
        "--mtp-quant", default=None,
        help=("MTP block precision (default: W8 with a W4 target, otherwise "
              "the target precision)"))
    args = parser.parse_args()
    convert_qwen3_moe(args.model_dir, args.output,
                      num_layers=args.layers,
                      prefill_seq_len=args.prefill_seq_len,
                      quant=args.quant,
                      mtp_quant=args.mtp_quant)
